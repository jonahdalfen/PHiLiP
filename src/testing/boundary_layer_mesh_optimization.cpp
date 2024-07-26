#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include "boundary_layer_mesh_optimization.hpp"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/mesh_error_estimate.h"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include "mesh/mesh_adaptation/mesh_optimizer.hpp"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BoundaryLayerMeshOptimization<dim, nstate> :: BoundaryLayerMeshOptimization(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
void BoundaryLayerMeshOptimization<dim, nstate> :: evaluate_regularization_matrix(
        dealii::TrilinosWrappers::SparseMatrix &regularization_matrix,
        std::shared_ptr<DGBase<dim,double>> dg) const
{
    (void) regularization_matrix;
    (void) dg;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    Parameters::AllParameters param_q1 = param;
    const bool use_oneD_parameteriation = true;
    int output_val = 0;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
    dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q1;
    evaluate_regularization_matrix(regularization_matrix_poisson_q1, flow_solver->dg);
    std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q1 = 
                std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param_q1, true);
    const bool output_refined_nodes = false;
    mesh_optimizer_q1->run_full_space_optimizer(regularization_matrix_poisson_q1, use_oneD_parameteriation, output_refined_nodes, output_val-1);
    output_vtk_files(flow_solver->dg, output_val-1);
}

template <int dim, int nstate>
double BoundaryLayerMeshOptimization<dim, nstate> ::output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int count_val) const
{
    (void) dg; (void) count_val;
    return 0;
}

template <int dim, int nstate>
int BoundaryLayerMeshOptimization<dim, nstate> :: run_test() const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    Parameters::AllParameters param_q1 = param;
    const bool use_oneD_parameteriation = true;
    int output_val = 0;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
    dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q1;
    evaluate_regularization_matrix(regularization_matrix_poisson_q1, flow_solver->dg);
    std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer_q1 = 
                std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param_q1, true);
    const bool output_refined_nodes = false;
    mesh_optimizer_q1->run_full_space_optimizer(regularization_matrix_poisson_q1, use_oneD_parameteriation, output_refined_nodes, output_val-1);
    output_vtk_files(flow_solver->dg, output_val-1);

    return 0;
}

#if PHILIP_DIM==2
template class BoundaryLayerMeshOptimization <PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    