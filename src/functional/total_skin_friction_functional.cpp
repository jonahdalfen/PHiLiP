
/* includes */

#include <iostream>
#include <vector>

#include "dg/dg.h"
#include "physics/physics.h"
#include "total_skin_friction_functional.hpp"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
TotalSkinFriction<dim,nstate,real,MeshType>::TotalSkinFriction(
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg, 
    const bool                                 /* _uses_solution_values*/,
    const bool                                 /*_uses_solution_gradient*/ )
    : Functional<dim,nstate,real,MeshType>(_dg)
{}

template<int dim, int nstate, typename real, typename MeshType>
    template<typename real2>
    real2 TotalSkinFriction<dim,nstate,real,MeshType>::evaluate_boundary_integrand(
        const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const dealii::Point<dim,real2> &/*phys_coord*/,
        const dealii::Tensor<1,dim,real2> &normal,
        const std::array<real2,nstate> &soln_at_q,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &soln_grad_at_q) const
        {
            if (boundary_id == 1001) 
            {
                const Physics::NavierStokes<dim,dim+2,real2> &navierstokes = dynamic_cast< const Physics::NavierStokes<dim,dim+2,real2> &> (physics);
                 dealii::Tensor<1,dim,double> drag_direction;
                 if constexpr (dim==2)
                 {
                    drag_direction[0] = cos(navierstokes.angle_of_attack);
                    drag_direction[1] = sin(navierstokes.angle_of_attack);
                 }
                 const std::array<real2,nstate> primitive_soln = navierstokes.convert_conservative_to_primitive_templated(soln_at_q); // from Euler
                 const std::array<dealii::Tensor<1,dim,real2>,nstate> primitive_soln_gradient
                 = navierstokes.convert_conservative_gradient_to_primitive_gradient_templated(soln_at_q,soln_grad_at_q);
                 real2 output_val = (navierstokes.compute_pressure(soln_at_q) * normal -(navierstokes.compute_viscous_stress_tensor(primitive_soln, primitive_soln_gradient) * normal)) * drag_direction;
                 return output_val;
            } 
            return 0.0;
        }
        
template class TotalSkinFriction <PHILIP_DIM, PHILIP_DIM+2, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalSkinFriction <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class TotalSkinFriction <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
}
