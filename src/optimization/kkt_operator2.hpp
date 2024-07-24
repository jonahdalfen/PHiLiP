#ifndef __KKT_OPERATOR_H__
#define __KKT_OPERATOR_H__

#include "ROL_Types.hpp"
#include "ROL_Objective.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_Vector_SimOpt.hpp"
#include "mesh/high_order_grid.h"
#include "rol_to_dealii_vector.hpp"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

/// KKT_Operator to be used with dealii::SolverBase class.
template<typename Real = double>
class KKT_Operator
{
protected:

    /// Objective function.
    const ROL::Ptr<ROL::Objective<Real>> objective_;
    /// Equality constraints.
    const ROL::Ptr<ROL::Constraint<Real>> equal_constraints_;

    /// Design variables.
    const ROL::Ptr<const ROL::Vector<Real>> design_variables_;
    /// Lagrange multipliers.
    const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult_;

    const std::shared_ptr<PHiLiP::HighOrderGrid<PHILIP_DIM,double>> high_order_grid;

    dealii::QGauss<PHILIP_DIM> quadrature;

    double min_cell_volume;


private:

    /// Used to perform the Lagrangian Hessian in two steps.
    const ROL::Ptr<ROL::Vector<Real>> temp_design_variables_size_vector_;
    /// Regularization parameter
    const Real regularization_parameter;

public:
    /// Constructor.
    KKT_Operator(
        const ROL::Ptr<ROL::Objective<Real>> objective,
        const ROL::Ptr<ROL::Constraint<Real>> equal_constraints,
        const ROL::Ptr<const ROL::Vector<Real>> design_variables,
        const ROL::Ptr<const ROL::Vector<Real>> lagrange_mult,
        const Real _regularization_parameter = 0.0)
        : objective_(objective)
        , equal_constraints_(equal_constraints)
        , design_variables_(design_variables)
        , lagrange_mult_(lagrange_mult)
        , high_order_grid(dynamic_cast<const PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equal_constraints).dg->high_order_grid)
        , quadrature(3)
        , temp_design_variables_size_vector_(design_variables->clone())
        , regularization_parameter(_regularization_parameter)
    { 
        double min_cell_volume_local = 1000000;
        dealii::FEValues<PHILIP_DIM, PHILIP_DIM> fe_values(*(high_order_grid->mapping_fe_field), high_order_grid->fe_system, quadrature, dealii::update_gradients | dealii::update_JxW_values);
        for(const auto &cell: high_order_grid->dof_handler_grid.active_cell_iterators())
        {
            if(! cell->is_locally_owned()) {continue;}

            fe_values.reinit(cell);

            double cell_volume = 0;

            for(unsigned int iquad = 0; iquad < fe_values.n_quadrature_points; ++iquad)
            {
                cell_volume += fe_values.JxW(iquad);
            }

            if(cell_volume < min_cell_volume_local)
            {
                min_cell_volume_local = cell_volume;
            }
        }

        min_cell_volume = dealii::Utilities::MPI::min(min_cell_volume_local, MPI_COMM_WORLD);
    }

    /// Returns the size of the KKT system.
    unsigned int size()
    {
        return (design_variables_->dimension() + lagrange_mult_->dimension());
    }

    /// Application of KKT matrix on vector src outputted into dst.
    void vmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        dst *= 0.0;

        static int number_of_times = 0;
        number_of_times++;
        std::cout << "Number of KKT vmult = " << number_of_times << std::endl;
        Real tol = 1e-15;
        const Real one = 1.0;

        ROL::Ptr<ROL::Vector<Real>> dst_rol = dst.getVector();
        const ROL::Ptr<const ROL::Vector<Real>> src_rol = src.getVector();

        auto &dst_split = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_rol);
        const auto &src_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_rol);

        ROL::Ptr<ROL::Vector<Real>> dst_design = dst_split.get_1();
        ROL::Ptr<ROL::Vector<Real>> dst_constraints = dst_split.get_2();
        dst_design->zero();
        dst_constraints->zero();

        const ROL::Ptr<const ROL::Vector<Real>> src_design = src_split.get_1();
        const ROL::Ptr<const ROL::Vector<Real>> src_constraints = src_split.get_2();

        // Top left block times top vector
        {
            objective_->hessVec(*dst_design, *src_design, *design_variables_, tol);
            equal_constraints_->applyAdjointHessian(*temp_design_variables_size_vector_, *lagrange_mult_, *src_design, *design_variables_, tol);
            dst_design->axpy(one, *temp_design_variables_size_vector_);

        }
        {
            // Add regularization parameter times identity
            ROL::Ptr<ROL::Vector<Real>> dst_control = dynamic_cast<ROL::Vector_SimOpt<Real>&>(*dst_design).get_2();
            const ROL::Ptr<const ROL::Vector<Real>> src_control = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*src_design).get_2();
            //dst_control->axpy(regularization_parameter, *src_control);
            regularization_vmult(*dst_control, *src_control);
            // Pretend Lagrangian Hessian is identity.
            //dst_design->set(*src_design);
        }

        // Top right block times bottom vector
        equal_constraints_->applyAdjointJacobian(*temp_design_variables_size_vector_, *src_constraints, *design_variables_, tol);
        dst_design->axpy(one, *temp_design_variables_size_vector_);

        // Bottom left left block times top vector
        equal_constraints_->applyJacobian(*dst_constraints, *src_design, *design_variables_, tol);

        // Bottom right block times bottom vector
        // 0 block in KKT
        dealii::deallog.depth_console(99);
    }

    /// Application of transposed KKT matrix on vector src outputted into dst.
    /** Same as vmult since KKT matrix is symmetric.
     */
    void Tvmult (dealiiSolverVectorWrappingROL<Real>       &dst,
                 const dealiiSolverVectorWrappingROL<Real> &src) const
    {
        vmult(dst, src);
    }

    /// Print the KKT system if the program is run with 1 MPI process.
    /** If more than 1 MPI process is used, we can't print out the matrix since
     *  the information is distributed
     */
    void print(const dealiiSolverVectorWrappingROL<Real> &vector_format)
    {
       const int do_full_matrix = (1 == dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
       std::cout << "do_full_matrix: " << do_full_matrix << std::endl;
       if (do_full_matrix) {
           dealiiSolverVectorWrappingROL<Real> column_of_kkt_operator;
           column_of_kkt_operator.reinit(vector_format);
           dealii::FullMatrix<double> fullA(vector_format.size());
           for (int i = 0; i < vector_format.size(); ++i) {
               std::cout << "COLUMN NUMBER: " << i+1 << " OUT OF " << vector_format.size() << std::endl;
               auto basis = vector_format.basis(i);
               MPI_Barrier(MPI_COMM_WORLD);
               this->vmult(column_of_kkt_operator,*basis);
               if (do_full_matrix) {
                   for (int j = 0; j < vector_format.size(); ++j) {
                       fullA[i][j] = column_of_kkt_operator[j];
                   }
               }
           }
           std::cout<<"Dense matrix:"<<std::endl;
           fullA.print_formatted(std::cout, 14, true, 10, "0", 1., 0.);
           std::abort();
       }
    }

    void regularization_vmult(ROL::Vector<double>& output_vector_rol, const ROL::Vector<double>& input_vector_rol) const
    {
        const auto &flow_constraint = dynamic_cast<const PHiLiP::FlowConstraints<PHILIP_DIM>&>(*equal_constraints_);
        const auto &input_vector_control = PHiLiP::ROL_vector_to_dealii_vector_reference(input_vector_rol);
        auto &output_vector_control = PHiLiP::ROL_vector_to_dealii_vector_reference(output_vector_rol);

        using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
        VectorType input_vector(high_order_grid->volume_nodes);
        VectorType output_vector(high_order_grid->volume_nodes);
        output_vector = 0;
        flow_constraint.dXvdXp.vmult(input_vector, input_vector_control);
        input_vector.update_ghost_values();
        dealii::FEValues<PHILIP_DIM, PHILIP_DIM> fe_values(*(high_order_grid->mapping_fe_field), high_order_grid->fe_system, quadrature, dealii::update_gradients | dealii::update_JxW_values);

        const unsigned int n_dofs_cell = high_order_grid->fe_system.dofs_per_cell;
        const unsigned int n_quad_points = fe_values.n_quadrature_points;
        std::vector<dealii::types::global_dof_index> dofs_indices(n_dofs_cell);
        for(const auto &cell: high_order_grid->dof_handler_grid.active_cell_iterators())
        {
            if(! cell->is_locally_owned()) {continue;}
            
            fe_values.reinit(cell);
            cell->get_dof_indices(dofs_indices);

            std::vector<std::array< dealii::Tensor<1,PHILIP_DIM,double>, PHILIP_DIM>> input_grad(n_quad_points);
            double cell_volume = 0;
            for(unsigned int iquad = 0; iquad < n_quad_points; ++iquad)
            {
                for(int istate = 0; istate < PHILIP_DIM; ++istate)
                {
                    input_grad[iquad][istate] = 0.0;
                }

                for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
                {
                    const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                    input_grad[iquad][istate] += input_vector(dofs_indices[idof]) * fe_values.shape_grad_component(idof, iquad, istate);
                }
                cell_volume += fe_values.JxW(iquad);
            }

            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int idof_global = dofs_indices[idof];
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;

                for(unsigned int iquad = 0; iquad < n_quad_points; ++iquad)
                {
                    output_vector(idof_global) += regularization_parameter*min_cell_volume/cell_volume * fe_values.shape_grad_component(idof, iquad, istate)*input_grad[iquad][istate]*fe_values.JxW(iquad);
                }
            }
        }
        output_vector.update_ghost_values();

        flow_constraint.dXvdXp.Tvmult_add(output_vector_control, output_vector);
        output_vector_control.update_ghost_values();
    }
};

#endif
