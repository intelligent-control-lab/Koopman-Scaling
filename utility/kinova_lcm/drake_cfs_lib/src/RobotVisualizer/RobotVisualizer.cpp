#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include "RobotVisualizer.h"

RobotVisualizer::RobotVisualizer(const std::string model_url)
{
    // connect plant and scene_graph
    std::tie(plant, scene_graph) = drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.001);
    drake::multibody::Parser parser(plant);

    // add robot environment
    parser.package_map().Add("project", "../../drake_cfs_lib");  
    parser.AddModelsFromUrl(model_url);
    plant->Finalize();

    // add meshcat
    meshcat = std::make_shared<drake::geometry::Meshcat>();
    drake::geometry::MeshcatVisualizerd::AddToBuilder(
        &builder, scene_graph->get_query_output_port(), meshcat);

    // add slider 
    Slider = builder.AddSystem<drake::multibody::meshcat::JointSliders<double>>(meshcat, plant);

    // build diagram
    diagram = builder.Build();

    cout << "RobotVisualizer is being create" << endl;
}

RobotVisualizer::~RobotVisualizer()
{
	cout << "RobotVisualizer is being deleted" << endl;
}

void RobotVisualizer::show_robot()
{
    diagram_context = diagram->CreateDefaultContext();  
    diagram->ForcedPublish(*diagram_context.get());
    Slider->Run(*diagram);
}

void RobotVisualizer::show_robot_trajectory(const Eigen::MatrixXd& joint_trajectory)
{
    diagram_context = diagram->CreateDefaultContext();  
    auto& plant_context = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
    // Eigen::MatrixXd M(7,7);
    for (int i = 0; i < joint_trajectory.rows(); i+=2) {
        plant->SetPositions(&plant_context, joint_trajectory.row(i));
        diagram->ForcedPublish(*diagram_context);
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// SDF RobotVisualizer::cal_minSDF_and_gradient(const Eigen::VectorXd& joint_position)
// {
//     SDF sign_distance_function;
//     double min_dist = 99999;

//     // get system contex
//     drake::systems::Context<double>& plant_context = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
//     drake::systems::Context<double>& context_scene_graph = diagram->GetMutableSubsystemContext(*scene_graph, diagram_context.get());

//     // if you need filter collision pair:
//     // drake::geometry::CollisionFilterManager copy_filter(scene_graph.collision_filter_manager());
//     // copy_filter = scene_graph.collision_filter_manager(context_scene_graph.get());

//     // get query_object
//     const drake::geometry::QueryObject<double>& query_object =
//         scene_graph->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(context_scene_graph);
//     const auto& inspector = query_object.inspector();

//     // set robot position
//     plant->SetPositions(&plant_context, joint_position);

//     //calculate sign distance function
//     drake::geometry::SignedDistancePair<double> min_signed_distance_pair;
//     const std::set<std::pair<drake::geometry::GeometryId, drake::geometry::GeometryId>>collision_candidate_pairs = query_object.inspector().GetCollisionCandidates();

//     for (const auto& geometry_pair:collision_candidate_pairs) 
//     {
//         drake::geometry::SignedDistancePair<double> signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(geometry_pair.first, geometry_pair.second);
//         if (signed_distance_pair.distance <= min_dist)
//         {
//           min_signed_distance_pair = signed_distance_pair;
//           min_dist = signed_distance_pair.distance;
//         }
//     }
//     bool verbose = false;
//     if (verbose == true)
//     {
//         cout << "min_dist:" << min_signed_distance_pair.distance << endl;
//         cout << "p_ACa:" << min_signed_distance_pair.p_ACa << endl;
//         cout << "id_A:" << inspector.GetName(inspector.GetFrameId(min_signed_distance_pair.id_A)) << endl;
//         cout << "p_BCb:" << min_signed_distance_pair.p_BCb << endl;
//         cout << "id_B:" << inspector.GetName(inspector.GetFrameId(min_signed_distance_pair.id_B)) << endl;
//         cout << "nhat_BA_W:" << min_signed_distance_pair.nhat_BA_W << endl;
//     }

//     // get sign distance info
//     Eigen::Vector3<double> p_ACa;
//     Eigen::Vector3<double> nhat_BA_W;
//     p_ACa = min_signed_distance_pair.p_ACa;
//     nhat_BA_W = min_signed_distance_pair.nhat_BA_W;

//     // calculate gradient of SDF
//     Eigen::Matrix3Xd Jq_v_BCa_W(3, plant->num_positions());
//     plant->CalcJacobianTranslationalVelocity(plant_context, drake::multibody::JacobianWrtVariable::kQDot,
//                                             plant->GetBodyFromFrameId(inspector.GetFrameId(min_signed_distance_pair.id_A))->body_frame(), p_ACa, plant->GetBodyFromFrameId(inspector.GetFrameId(min_signed_distance_pair.id_B))->body_frame(),
//                                             plant->world_frame(),&Jq_v_BCa_W);
//     Eigen::VectorXd ddistance_dq(plant->num_positions());
//     ddistance_dq = nhat_BA_W.transpose() * Jq_v_BCa_W;

//     sign_distance_function.min_distance = min_dist;
//     sign_distance_function.ddistance_dq = ddistance_dq;

//     return sign_distance_function;
// }