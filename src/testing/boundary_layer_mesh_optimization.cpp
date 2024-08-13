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

template<int dim, int nstate>
void BoundaryLayerMeshOptimization<dim,nstate>::write_solution_volume_nodes_to_file(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const int n_cells = dg->triangulation->n_global_active_cells();
    const int poly_degree = dg->get_min_fe_degree();
    const std::string filename_soln = 
                    "solution_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const std::string filename_volnodes = 
                    "volnodes_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const dealii::IndexSet &soln_range = dg->solution.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    std::ofstream outfile_soln(filename_soln);
    std::ofstream outfile_volnodes(filename_volnodes);

    if( (!outfile_soln.is_open()) || (!outfile_volnodes.is_open()))
    {
        std::cout<<"Could not open file. Aborting.."<<std::endl;
        std::abort();
    }

    for(const auto &isol : soln_range)
    {
        outfile_soln<<std::setprecision(16)<<dg->solution(isol)<<"\n";
    }
    for(const auto &ivol : vol_range)
    {
        outfile_volnodes<<std::setprecision(16)<<dg->high_order_grid->volume_nodes(ivol)<<"\n";
    }
    outfile_soln.close();
    outfile_volnodes.close();
}

template<int dim, int nstate>
void BoundaryLayerMeshOptimization<dim,nstate>::read_solution_volume_nodes_from_file(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const int n_cells = dg->triangulation->n_global_active_cells();
    const int poly_degree = dg->get_min_fe_degree();
    const std::string filename_soln = 
                    "solution_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const std::string filename_volnodes = 
                    "volnodes_" + std::to_string(this->mpi_rank) + "_cells" + std::to_string(n_cells) + "_p" + std::to_string(poly_degree);
    const dealii::IndexSet &soln_range = dg->solution.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    std::ifstream infile_soln(filename_soln);
    std::ifstream infile_volnodes(filename_volnodes);

    if( (!infile_soln.is_open()) || (!infile_volnodes.is_open()))
    {
        std::cout<<"Could not open file. Aborting.."<<std::endl;
        std::abort();
    }

    for(const auto &isol : soln_range)
    {
        infile_soln>>dg->solution(isol);
    }
    for(const auto &ivol : vol_range)
    {
        infile_volnodes>>dg->high_order_grid->volume_nodes(ivol);
    }
    infile_soln.close();
    infile_volnodes.close();
    dg->high_order_grid->volume_nodes.update_ghost_values();
    dg->solution.update_ghost_values();
}

template <int dim, int nstate>
void BoundaryLayerMeshOptimization<dim, nstate> :: evaluate_regularization_matrix(
        dealii::TrilinosWrappers::SparseMatrix &regularization_matrix,
        std::shared_ptr<DGBase<dim,double>> dg) const
{
    // Get volume of smallest element.
    const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[dg->high_order_grid->grid_degree];
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_vol(mapping, dg->high_order_grid->fe_metric_collection[dg->high_order_grid->grid_degree], volume_quadrature,
                    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_vol.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_values_vol.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_vol.dofs_per_cell);
    
    double min_cell_volume_local = 1.0e6;
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        double cell_vol = 0.0;
        fe_values_vol.reinit (cell);


        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }


        if(cell_vol < min_cell_volume_local)
        {
            min_cell_volume_local = cell_vol;
        }
    }


    const double min_cell_vol = dealii::Utilities::MPI::min(min_cell_volume_local, mpi_communicator);


    // Set sparsity pattern
    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dg->high_order_grid->dof_handler_grid,
                                            hanging_node_constraints);
    hanging_node_constraints.close();


    dealii::DynamicSparsityPattern dsp(dg->high_order_grid->dof_handler_grid.n_dofs(), dg->high_order_grid->dof_handler_grid.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dg->high_order_grid->dof_handler_grid, dsp, hanging_node_constraints);
    const dealii::IndexSet &locally_owned_dofs = dg->high_order_grid->locally_owned_dofs_grid;
    regularization_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, this->mpi_communicator);


    // Set elements.
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_vol.reinit (cell);
        cell->get_dof_indices(dofs_indices);
        cell_matrix = 0;
        
        double cell_vol = 0.0;
        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }
        const double omega_k = min_cell_vol/cell_vol;


        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int icomp = fe_values_vol.get_fe().system_to_component_index(i).first;
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int jcomp = fe_values_vol.get_fe().system_to_component_index(j).first;
                double val_ij = 0.0;


                if(icomp == jcomp)
                {
                    for(unsigned int q=0; q<n_quad_pts; ++q)
                    {
                        val_ij += omega_k*fe_values_vol.shape_grad(i,q)*fe_values_vol.shape_grad(j,q)*fe_values_vol.JxW(q);
                    }
                }
                cell_matrix(i,j) = val_ij;
            }
        }
        hanging_node_constraints.distribute_local_to_global(cell_matrix, dofs_indices, regularization_matrix); 
    } // cell loop ends
    regularization_matrix.compress(dealii::VectorOperation::add);
}

template <int dim, int nstate>
double BoundaryLayerMeshOptimization<dim, nstate> ::output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int count_val) const
{
    const int outputval = 7000 + count_val;
    dg->output_results_vtk(outputval);

    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    dg->assemble_residual();
    return abs_dwr_error;

    return 0;
}

template <int dim, int nstate>
int BoundaryLayerMeshOptimization<dim, nstate> :: run_test() const
{
    std::cout<<"Running the optimization case with p0."<<std::endl;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    Parameters::AllParameters param_q1 = param;
    int output_val = 0;
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
    output_vtk_files(flow_solver->dg, ++output_val);

/*
    dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson;
    evaluate_regularization_matrix(regularization_matrix_poisson, flow_solver->dg);
    std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer = 
                std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param_q1, true);

    const bool output_refined_nodes = false;
    const bool use_oneD_parameteriation = true;

    
    mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson, use_oneD_parameteriation, output_refined_nodes, output_val);
   std::cout<<"Completed optimization with p0."<<std::endl;
   flow_solver->run();
*/
    // write_solution_volume_nodes_to_file(flow_solver->dg);

    // read_solution_volume_nodes_from_file(flow_solver->dg);
    output_vtk_files(flow_solver->dg, output_val+1);
  
    // std::cout<<"Interpolating grid from q1 to q2."<<std::endl;
    // const unsigned int grid_degree_updated = 2;
    // flow_solver->dg->high_order_grid->set_q_degree(grid_degree_updated, true);
    /*
    output_vtk_files(flow_solver->dg, output_val+1);
    dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2;
    evaluate_regularization_matrix(regularization_matrix_poisson_q2, flow_solver->dg);
    mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson_q2, use_oneD_parameteriation, output_refined_nodes, output_val);
    output_vtk_files(flow_solver->dg, output_val+1);
    write_solution_volume_nodes_to_file(flow_solver->dg);
    */
    // Refine and interpolate the mesh
    // read_solution_volume_nodes_from_file(flow_solver->dg);
    /*
    std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation = std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));
    meshadaptation->adapt_mesh();
    dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson_q2_refined;
    evaluate_regularization_matrix(regularization_matrix_poisson_q2_refined, flow_solver->dg);
    mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson_q2_refined, use_oneD_parameteriation, output_refined_nodes, output_val);
    */
    // output_vtk_files(flow_solver->dg, ++output_val);
    //write_solution_volume_nodes_to_file(flow_solver->dg);




    // flow_solver->dg->set_p_degree_and_interpolate_solution(1);
    // mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson, use_oneD_parameteriation, output_refined_nodes, output_val);
    // flow_solver->run();
    // std::cout<<"Completed optimization with p1. Outputting file with converged flow..."<<std::endl;

    // output_vtk_files(flow_solver->dg, output_val+1);

    return 0;
}



#if PHILIP_DIM==2
template class BoundaryLayerMeshOptimization <PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
