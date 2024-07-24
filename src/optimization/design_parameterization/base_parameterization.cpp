#include "base_parameterization.hpp"

namespace PHiLiP {

template<int dim>
BaseParameterization<dim> :: BaseParameterization (
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : high_order_grid(_high_order_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi);
}

template<int dim>
void BaseParameterization<dim> :: output_design_variables(unsigned int /*iteration_no*/) const
{
    // Outputs nothing by default. Overriden in derived classes.
}

template<int dim>
void BaseParameterization<dim> ::  output_control_nodes(const std::string /*filename*/) const
{
}
template<int dim>
void BaseParameterization<dim> ::  output_control_nodes_with_interpolated_high_order_nodes() const
{
}
template<int dim>
void BaseParameterization<dim> ::  output_control_nodes_refined() const
{
}

template<int dim>
bool BaseParameterization<dim> :: has_design_variable_been_updated(
    const VectorType &current_design_var, 
    const VectorType &updated_design_var) const
{
    VectorType diff = current_design_var;
    diff -= updated_design_var;
    diff.update_ghost_values();
    
    bool is_design_variable_updated = true;
    if(diff.l2_norm() == 0.0) {is_design_variable_updated = false;}
    
    return is_design_variable_updated;
}

template<int dim>
int BaseParameterization<dim> :: is_design_variable_valid(
    const MatrixType & /*dXv_dXp*/, 
    const VectorType & /*design_var*/) const
{
    return 0;
}

template class BaseParameterization<PHILIP_DIM>;
} // PHiLiP namespace
