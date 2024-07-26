#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check the goal-oriented mesh adaptation locations for various manufactured solutions.
template <int dim, int nstate>
class BoundaryLayerMeshOptimization : public TestsBase
{
public:
    /// Constructor of BoundaryLayerMeshOptimization.
    BoundaryLayerMeshOptimization(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test to check the location of refined cell after performing goal-oriented mesh adaptation.
    int run_test() const;

    /// Outputs vtk files with primal and adjoint solutions.
    double output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int count_val) const;

    void evaluate_regularization_matrix(
        dealii::TrilinosWrappers::SparseMatrix &regularization_matrix,
        std::shared_ptr<DGBase<dim,double>> dg) const;

}; 

} // Tests namespace
} // PHiLiP namespace

