#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include "KinematicTrajOpt.h"

KinematicTrajOpt::KinematicTrajOpt(const std::string model_url)
{
    // connect plant and scene_graph
    std::tie(plant, scene_graph) = drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.001);
    drake::multibody::Parser parser(plant);

    // add robot environment
    parser.package_map().Add("project", "../../drake_cfs_lib");  
    parser.AddModelsFromUrl(model_url);
    plant->Finalize();
    diagram = builder.Build();
    cout << "KinematicTrajOpt is being create" << endl;
}

KinematicTrajOpt::~KinematicTrajOpt()
{
	cout << "KinematicTrajOpt is being deleted" << endl;
}

TrajOptResult KinematicTrajOpt::solve_trajectory(const Eigen::VectorXd& start_position,const Eigen::VectorXd& goal_position)
{
    TrajOptResult traj_opt_result;
    drake::planning::trajectory_optimization::KinematicTrajectoryOptimization trajopt(plant->num_positions(),10);
    trajopt.AddDurationCost(1.0);
    trajopt.AddPathLengthCost(1.0);
    trajopt.AddPositionBounds(plant->GetPositionLowerLimits(), plant->GetPositionUpperLimits());
    trajopt.AddVelocityBounds(plant->GetVelocityLowerLimits(), plant->GetVelocityUpperLimits());
    trajopt.AddDurationConstraint(0.5, 5);
    trajopt.AddPathPositionConstraint(start_position, start_position, 0);
    trajopt.AddPathPositionConstraint(goal_position, goal_position, 1);
    trajopt.AddPathVelocityConstraint(Eigen::VectorXd::Zero(plant->num_positions()), Eigen::VectorXd::Zero(plant->num_positions()), 0);  
    trajopt.AddPathVelocityConstraint(Eigen::VectorXd::Zero(plant->num_positions()), Eigen::VectorXd::Zero(plant->num_positions()), 1);  

    // solve
    auto result = drake::solvers::Solve(trajopt.prog());
    if (result.is_success()) 
    {
        // Reconstruct the trajectory
        auto q = trajopt.ReconstructTrajectory(result);

        // add the collision constraint for second optimization
        diagram_context = diagram->CreateDefaultContext();  
        drake::systems::Context<double>& plant_context = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
        auto collision_constraint = std::make_shared<drake::multibody::MinimumDistanceLowerBoundConstraint>(
        plant, 0.0015, &plant_context,drake::solvers::QuadraticallySmoothedHingeLoss, 0.001);
        Eigen::VectorXd evaluate_at_s = Eigen::VectorXd::LinSpaced(30, 0, 1);
        for (int i = 0; i < evaluate_at_s.size(); ++i) {
            double s = evaluate_at_s(i);
            trajopt.AddPathPositionConstraint(collision_constraint, s);
        }
        trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result));

        auto finial_result = drake::solvers::Solve(trajopt.prog());
        if (finial_result.is_success()) 
        {
            auto q = trajopt.ReconstructTrajectory(result);
            auto qdot = q.MakeDerivative();
            double start_time = q.start_time();
            double end_time = q.end_time();
            double time_step = 0.001; 
            int num_steps = static_cast<int>((end_time - start_time) / time_step) + 1;
            traj_opt_result.trajectory.resize(num_steps, 7);
            //Iterate over time steps from start_time to end_time
            int step_idx = 0;
            for (double t = start_time; t <= end_time; t += time_step) {
                traj_opt_result.trajectory.row(step_idx) = q.value(t).transpose();
                step_idx++;
            }
            // Set last trajectory to the goal position
            traj_opt_result.trajectory.row(traj_opt_result.trajectory.rows()-1) = q.value(end_time).transpose();
            traj_opt_result.is_success = true;
            return traj_opt_result;
        }
        else 
        {
            traj_opt_result.is_success = false;
            std::cerr << "Final Optimization failed!" << std::endl;
        }
    } else 
    {
        traj_opt_result.is_success = false;
        std::cerr << "Initial Optimization failed!" << std::endl;
    }
}

SDF KinematicTrajOpt::cal_minSDF_and_gradient(const Eigen::VectorXd& joint_position)
{
    SDF sign_distance_function;
    double min_dist = 99999;

    // get system contex
    drake::systems::Context<double>& plant_context = diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
    drake::systems::Context<double>& context_scene_graph = diagram->GetMutableSubsystemContext(*scene_graph, diagram_context.get());

    // if you need filter collision pair:
    // drake::geometry::CollisionFilterManager copy_filter(scene_graph.collision_filter_manager());
    // copy_filter = scene_graph.collision_filter_manager(context_scene_graph.get());

    // get query_object
    const drake::geometry::QueryObject<double>& query_object =
        scene_graph->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(context_scene_graph);
    const auto& inspector = query_object.inspector();

    // set robot position
    plant->SetPositions(&plant_context, joint_position);

    //calculate sign distance function
    drake::geometry::SignedDistancePair<double> min_signed_distance_pair;
    const std::set<std::pair<drake::geometry::GeometryId, drake::geometry::GeometryId>>collision_candidate_pairs = query_object.inspector().GetCollisionCandidates();

    for (const auto& geometry_pair:collision_candidate_pairs) 
    {
        drake::geometry::SignedDistancePair<double> signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(geometry_pair.first, geometry_pair.second);
        if (signed_distance_pair.distance <= min_dist)
        {
          min_signed_distance_pair = signed_distance_pair;
          min_dist = signed_distance_pair.distance;
        }
    }
    bool verbose = false;
    if (verbose == true)
    {
        cout << "min_dist:" << min_signed_distance_pair.distance << endl;
        cout << "p_ACa:" << min_signed_distance_pair.p_ACa << endl;
        cout << "id_A:" << inspector.GetName(inspector.GetFrameId(min_signed_distance_pair.id_A)) << endl;
        cout << "p_BCb:" << min_signed_distance_pair.p_BCb << endl;
        cout << "id_B:" << inspector.GetName(inspector.GetFrameId(min_signed_distance_pair.id_B)) << endl;
        cout << "nhat_BA_W:" << min_signed_distance_pair.nhat_BA_W << endl;
    }

    // get sign distance info
    Eigen::Vector3d p_ACa;
    Eigen::Vector3d nhat_BA_W;
    p_ACa = min_signed_distance_pair.p_ACa;
    nhat_BA_W = min_signed_distance_pair.nhat_BA_W;

    // calculate gradient of SDF
    Eigen::Matrix3Xd Jq_v_BCa_W(3, plant->num_positions());
    plant->CalcJacobianTranslationalVelocity(plant_context, drake::multibody::JacobianWrtVariable::kQDot,
                                            plant->GetBodyFromFrameId(inspector.GetFrameId(min_signed_distance_pair.id_A))->body_frame(), p_ACa, plant->GetBodyFromFrameId(inspector.GetFrameId(min_signed_distance_pair.id_B))->body_frame(),
                                            plant->world_frame(),&Jq_v_BCa_W);
    Eigen::VectorXd ddistance_dq(plant->num_positions());
    ddistance_dq = nhat_BA_W.transpose() * Jq_v_BCa_W;

    sign_distance_function.min_distance = min_dist;
    sign_distance_function.ddistance_dq = ddistance_dq;

    return sign_distance_function;
}