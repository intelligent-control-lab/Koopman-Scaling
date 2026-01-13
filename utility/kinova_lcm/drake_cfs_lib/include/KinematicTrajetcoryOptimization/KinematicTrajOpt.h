#ifndef __KINE_TRAJ_OPT_H_
#define __KINE_TRAJ_OPT_H_

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <drake/common/drake_assert.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/basic_vector.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/continuous_state.h>
#include <drake/systems/framework/leaf_system.h>
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_animation.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/multibody/parsing/model_directives.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/query_object.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/meshcat/joint_sliders.h"
#include "drake/systems/framework/system.h"
#include "drake/multibody/tree/fixed_offset_frame.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/planning/trajectory_optimization/kinematic_trajectory_optimization.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_lower_bound_constraint.h"

using namespace std;
struct SDF 
{
   double min_distance;
   Eigen::VectorXd ddistance_dq;
};

struct TrajOptResult
{
   double is_success;
   Eigen::MatrixXd trajectory;
};

class KinematicTrajOpt{
   public:
	  	KinematicTrajOpt(const std::string model_url);
	  	~KinematicTrajOpt();
      SDF cal_minSDF_and_gradient(const Eigen::VectorXd& joint_position);
      TrajOptResult solve_trajectory(const Eigen::VectorXd& start_position,const Eigen::VectorXd& goal_position);

   private:
      drake::systems::DiagramBuilder<double> builder;
      std::unique_ptr<drake::systems::Diagram<double>> diagram;
      drake::multibody::MultibodyPlant<double>* plant{};
      drake::geometry::SceneGraph<double>* scene_graph{};
      std::unique_ptr<drake::systems::Context<double>> diagram_context;
};

#endif  // __KINE_TRAJ_OPT_H_