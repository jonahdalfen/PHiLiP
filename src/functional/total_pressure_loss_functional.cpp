
/* includes */

#include <iostream>
#include <vector>

#include "dg/dg_base.hpp"
#include "physics/physics.h"
#include "total_pressure_loss_functional.hpp"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
TotalPressureLoss<dim,nstate,real,MeshType>::TotalPressureLoss(
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg, 
    const bool                                 /* _uses_solution_values*/,
    const bool                                 /*_uses_solution_gradient*/ )
    : Functional<dim,nstate,real,MeshType>(_dg)
{}

template<int dim, int nstate, typename real, typename MeshType>
    template<typename real2>
    real2 TotalPressureLoss<dim,nstate,real,MeshType>::evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const dealii::Tensor<1,dim,real2> &/*normal*/,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
        {
            if (boundary_id == 1002) 
            {
                const Physics::Euler<dim,dim+2,real2> &euler = dynamic_cast< const Physics::Euler<dim,dim+2,real2> &> (physics);
                return euler.compute_pressure(soln_at_q);
            } 
            else if (boundary_id == 1003) 
            {
                const Physics::Euler<dim,dim+2,real2> &euler = dynamic_cast< const Physics::Euler<dim,dim+2,real2> &> (physics);
                return -euler.compute_pressure(soln_at_q);
            }
            return 0.0;
        }
        
template class TotalPressureLoss <PHILIP_DIM, PHILIP_DIM+2, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalPressureLoss <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class TotalPressureLoss <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
}
