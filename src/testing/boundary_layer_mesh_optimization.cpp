#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include "boundary_layer_mesh_optimization.hpp"
#include "flow_solver/flow_solver_factory.h"

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
int BoundaryLayerMeshOptimization<dim, nstate> :: run_test () const
{
 return 0;
}

#if PHILIP_DIM==2
template class BoundaryLayerMeshOptimization <PHILIP_DIM, PHILIP_DIM+2>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    