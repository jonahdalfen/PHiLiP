#include "periodic_turbulence.h"

#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include "math.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// TURBULENCE IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
PeriodicTurbulence<dim, nstate>::PeriodicTurbulence(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_param.flow_solver_param.flow_case_type;

    // Flow case identifiers
    this->is_taylor_green_vortex = (flow_type == FlowCaseEnum::taylor_green_vortex);
    this->is_viscous_flow = (this->all_param.pde_type != Parameters::AllParameters::PartialDifferentialEquation::euler);
    this->do_calculate_numerical_entropy= this->all_param.flow_solver_param.do_calculate_numerical_entropy;

    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = Parameters::AllParameters::PartialDifferentialEquation::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));

    /* Initialize integrated quantities as NAN; 
       done as a precaution in the case compute_integrated_quantities() is not called
       before a member function of kind get_integrated_quantity() is called
     */
    std::fill(this->integrated_quantities.begin(), this->integrated_quantities.end(), NAN);
}

template <int dim, int nstate>
void PeriodicTurbulence<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    std::string flow_type_string;
    if(this->is_taylor_green_vortex) {
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
        this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    }
    this->display_grid_parameters();
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    if(this->all_param.flow_solver_param.constant_time_step > 0.0) {
        const double constant_time_step = this->all_param.flow_solver_param.constant_time_step;
        this->pcout << "- - Using constant time step in FlowSolver parameters: " << constant_time_step << std::endl;
        return constant_time_step;
    } else {
        const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
        const double constant_time_step = this->all_param.flow_solver_param.courant_friedrich_lewy_number * approximate_grid_spacing;
        return constant_time_step;
    }
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double cfl_number = this->all_param.flow_solver_param.courant_friedrich_lewy_number;
    const double time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;
    return time_step;
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_and_update_integrated_quantities(DGBase<dim, double> &dg)
{
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);
    
    // Initialize the maximum local wave speed to zero; only used for adaptive time step
    if(this->all_param.flow_solver_param.adaptive_time_step == true) this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (int s=0; s<nstate; ++s) {
                for (int d=0; d<dim; ++d) {
                    soln_grad_at_q[s][d] = 0.0;
                }
            }
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                soln_grad_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_grad_component(idof,iquad,istate);
            }
            // const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrand_values;
            std::fill(integrand_values.begin(), integrand_values.end(), 0.0);
            integrand_values[IntegratedQuantitiesEnum::kinetic_energy] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(soln_at_q);
            integrand_values[IntegratedQuantitiesEnum::enstrophy] = this->navier_stokes_physics->compute_enstrophy(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::pressure_dilatation] = this->navier_stokes_physics->compute_pressure_dilatation(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr] = this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);

            for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
                integral_values[i_quantity] += integrand_values[i_quantity] * fe_values_extra.JxW(iquad);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue) if using an adaptive time step
            if(this->all_param.flow_solver_param.adaptive_time_step == true) {
                const double local_wave_speed = this->navier_stokes_physics->max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
            }
        }
    }

    // update integrated quantities
    for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
        this->integrated_quantities[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
        this->integrated_quantities[i_quantity] /= this->domain_size; // divide by total domain volume
    }
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_kinetic_energy() const
{
    const double integrated_kinetic_energy = this->integrated_quantities[IntegratedQuantitiesEnum::kinetic_energy];
    // // Abort if energy is nan
    // if(std::isnan(integrated_kinetic_energy)) {
    //     this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
    //     this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
    //     std::abort();
    // }
    return integrated_kinetic_energy;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_enstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_vorticity_based_dissipation_rate() const
{
    const double integrated_enstrophy = this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
    double vorticity_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        vorticity_based_dissipation_rate = this->navier_stokes_physics->compute_vorticity_based_dissipation_rate_from_integrated_enstrophy(integrated_enstrophy);
    }
    return vorticity_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_pressure_dilatation_based_dissipation_rate() const
{
    const double integrated_pressure_dilatation = this->integrated_quantities[IntegratedQuantitiesEnum::pressure_dilatation];
    return (-1.0*integrated_pressure_dilatation); // See reference (listed in header file), equation (57b)
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_deviatoric_strain_rate_tensor_based_dissipation_rate() const
{
    const double integrated_deviatoric_strain_rate_tensor_magnitude_sqr = this->integrated_quantities[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr];
    double deviatoric_strain_rate_tensor_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        deviatoric_strain_rate_tensor_based_dissipation_rate = 
            this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr(integrated_deviatoric_strain_rate_tensor_magnitude_sqr);
    }
    return deviatoric_strain_rate_tensor_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_numerical_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    const double poly_degree = this->all_param.flow_solver_param.poly_degree;
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(this->all_param.use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);


    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            std::array<double,nstate> entropy_var;
            const double density = soln_state[0];
            dealii::Tensor<1,dim,double> vel;
            double vel2 = 0.0;
            for(int idim=0; idim<dim; idim++){
                vel[idim] = soln_state[idim+1]/soln_state[0];
                vel2 += vel[idim]*vel[idim];
            }
            const double pressure = 0.4*(soln_state[nstate-1] - 0.5*density*vel2);
            const double entropy = log(pressure) - 1.4 * log(density);
            
            entropy_var[0] = (1.4-entropy)/0.4 - 0.5 * density / pressure * vel2;
            for(int idim=0; idim<dim; idim++){
                entropy_var[idim+1] = soln_state[idim+1] / pressure;
            }
            entropy_var[nstate-1] = - density / pressure;

            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
                                                
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }

    //Note that dot product accounts for MPI distributed vectors
    //Therefore, there is no need for an MPI sum.
    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    return entropy;
}

template <int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Compute and update integrated quantities
    this->compute_and_update_integrated_quantities(*dg);
    // Get computed quantities
    const double integrated_kinetic_energy = this->get_integrated_kinetic_energy();
    const double integrated_enstrophy = this->get_integrated_enstrophy();
    const double vorticity_based_dissipation_rate = this->get_vorticity_based_dissipation_rate();
    const double pressure_dilatation_based_dissipation_rate = this->get_pressure_dilatation_based_dissipation_rate();
    const double deviatoric_strain_rate_tensor_based_dissipation_rate = this->get_deviatoric_strain_rate_tensor_based_dissipation_rate();
    
    double numerical_entropy = 0;
    if (do_calculate_numerical_entropy) numerical_entropy = this->get_numerical_entropy(dg);

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        if(do_calculate_numerical_entropy) this->add_value_to_data_table(numerical_entropy,"numerical_entropy",unsteady_data_table);
        this->add_value_to_data_table(integrated_kinetic_energy,"kinetic_energy",unsteady_data_table);
        this->add_value_to_data_table(integrated_enstrophy,"enstrophy",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(vorticity_based_dissipation_rate,"eps_vorticity",unsteady_data_table);
        this->add_value_to_data_table(pressure_dilatation_based_dissipation_rate,"eps_pressure",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(deviatoric_strain_rate_tensor_based_dissipation_rate,"eps_strain",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Energy: " << integrated_kinetic_energy
                << "    Enstrophy: " << integrated_enstrophy;
    if(is_viscous_flow) {
        this->pcout << "    eps_vorticity: " << vorticity_based_dissipation_rate
                    << "    eps_p+eps_strain: " << (pressure_dilatation_based_dissipation_rate + deviatoric_strain_rate_tensor_based_dissipation_rate);
    }
    if(do_calculate_numerical_entropy){
        this->pcout << "    Num. Entropy: " << std::setprecision(16) << numerical_entropy;
    }
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(integrated_kinetic_energy)) {
        this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

#if PHILIP_DIM==3
template class PeriodicTurbulence <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

