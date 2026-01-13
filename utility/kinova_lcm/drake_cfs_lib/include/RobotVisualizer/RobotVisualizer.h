#ifndef __ROBOT_MODEL_H_
#define __ROBOT_MODEL_H_

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

using namespace std;

class RobotVisualizer{
   public:
	  	RobotVisualizer(const std::string model_url);
	  	~RobotVisualizer();
      void show_robot();
      void show_robot_trajectory(const Eigen::MatrixXd& joint_trajectory);
      drake::multibody::MultibodyPlant<double>* plant{};
      
   private:
      drake::systems::DiagramBuilder<double> builder;
      std::unique_ptr<drake::systems::Diagram<double>> diagram;
      std::shared_ptr<drake::geometry::Meshcat> meshcat;
      drake::multibody::meshcat::JointSliders<double>* Slider; 
      drake::geometry::SceneGraph<double>* scene_graph{};
      std::unique_ptr<drake::systems::Context<double>> diagram_context;
};

#endif  // __ROBOT_MODEL_H_