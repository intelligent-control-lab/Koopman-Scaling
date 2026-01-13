/*
* KINOVA (R) KORTEX (TM)
*
* Copyright (c) 2018 Kinova inc. All rights reserved.
*
* This software may be modified and distributed
* under the terms of the BSD 3-Clause license.
*
* Refer to the LICENSE file for details.
*
*/
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

#include <KDetailedException.h>

#include <BaseClientRpc.h>
#include <BaseCyclicClientRpc.h>
#include <SessionClientRpc.h>
#include <SessionManager.h>

#include <RouterClient.h>
#include <TransportClientTcp.h>
#include <TransportClientUdp.h>

#include <google/protobuf/util/json_util.h>

#include "kinova_interface.hpp"
#include <lcm/lcm-cpp.hpp>

#include "utilities.h"

#if defined(_MSC_VER)
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <time.h>
#include <fstream> // Added for file output

namespace k_api = Kinova::Api;

constexpr auto TIMEOUT_DURATION = std::chrono::seconds{20};

#define PORT 10000
#define PORT_REAL_TIME 10001

#define DURATION 5             // Network timeout (seconds)

float velocity = 20.0f;         // Default velocity of the actuator (degrees per seconds)
float time_duration = DURATION; // Duration of the example (seconds)

// Waiting time during actions
const auto ACTION_WAITING_TIME = std::chrono::seconds(1);

// Create an event listener that will set the promise action event to the exit value
// Will set promise to either END or ABORT
// Use finish_promise.get_future.get() to wait and get the value
std::function<void(k_api::Base::ActionNotification)> 
    create_event_listener_by_promise(std::promise<k_api::Base::ActionEvent>& finish_promise)
{
    return [&finish_promise] (k_api::Base::ActionNotification notification)
    {
        const auto action_event = notification.action_event();
        switch(action_event)
        {
        case k_api::Base::ActionEvent::ACTION_END:
        case k_api::Base::ActionEvent::ACTION_ABORT:
            finish_promise.set_value(action_event);
            break;
        default:
            break;
        }
    };
}

// Create closure to set finished to true after an END or an ABORT
std::function<void(k_api::Base::ActionNotification)> 
check_for_end_or_abort(bool& finished)
{
    return [&finished](k_api::Base::ActionNotification notification)
    {
        std::cout << "EVENT : " << k_api::Base::ActionEvent_Name(notification.action_event()) << std::endl;

        // The action is finished when we receive a END or ABORT event
        switch(notification.action_event())
        {
        case k_api::Base::ActionEvent::ACTION_ABORT:
        case k_api::Base::ActionEvent::ACTION_END:
            finished = true;
            break;
        default:
            break;
        }
    };
}

/*****************************
 * Example related function *
 *****************************/
int64_t GetTickUs()
{
#if defined(_MSC_VER)
    LARGE_INTEGER start, frequency;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    return (start.QuadPart * 1000000)/frequency.QuadPart;
#else
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    return (start.tv_sec * 1000000LLU) + (start.tv_nsec / 1000);
#endif
}

/**************************
 * Example core functions *
 **************************/
void example_move_to_home_position(k_api::Base::BaseClient* base)
{
    // Make sure the arm is in Single Level Servoing before executing an Action
    auto servoingMode = k_api::Base::ServoingModeInformation();
    servoingMode.set_servoing_mode(k_api::Base::ServoingMode::SINGLE_LEVEL_SERVOING);
    base->SetServoingMode(servoingMode);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Move arm to ready position
    std::cout << "Moving the arm to a safe position" << std::endl;
    auto action_type = k_api::Base::RequestedActionType();
    action_type.set_action_type(k_api::Base::REACH_JOINT_ANGLES);
    auto action_list = base->ReadAllActions(action_type);
    auto action_handle = k_api::Base::ActionHandle();
    action_handle.set_identifier(0);
    for (auto action : action_list.action_list()) 
    {
        if (action.name() == "Home") 
        {
            action_handle = action.handle();
        }
    }

    if (action_handle.identifier() == 0) 
    {
        std::cout << "Can't reach safe position, exiting" << std::endl;
    } 
    else 
    {
        bool action_finished = false; 
        // Notify of any action topic event
        auto options = k_api::Common::NotificationOptions();
        auto notification_handle = base->OnNotificationActionTopic(
            check_for_end_or_abort(action_finished),
            options
        );

        base->ExecuteActionFromReference(action_handle);

        while(!action_finished)
        { 
            std::this_thread::sleep_for(ACTION_WAITING_TIME);
        }

        base->Unsubscribe(notification_handle);
    }
}

double g_joint_pos_des[7];

double joint_fbk_pos[7];
double joint_fbk_spd[7];
double joint_fbk_trq[7];

class Handler 
{
    public:
        ~Handler() {}

        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const exlcm::kinova_interface* msg)
        {
            for (int i =0;i<7;i++)
            {
                g_joint_pos_des[i] = msg->joint_des_pos[i];
            }
        }
};

bool LCM_test(k_api::Base::BaseClient* base, k_api::BaseCyclic::BaseCyclicClient* base_cyclic)
{
    bool return_status = true;

    // Move arm to ready position
    // example_move_to_home_position(base);
    
    k_api::BaseCyclic::Feedback base_feedback;
    k_api::BaseCyclic::Command  base_command;

    std::vector<float> commands;

    auto servoingMode = k_api::Base::ServoingModeInformation();

    int timer_count = 0;
    int64_t now = 0;
    int64_t last = 0;

    int timeout = 0;

    std::cout << "Initializing the arm for velocity low-level control example" << std::endl;
    try
    {
        // Set the base in low-level servoing mode
        servoingMode.set_servoing_mode(k_api::Base::ServoingMode::LOW_LEVEL_SERVOING);
        base->SetServoingMode(servoingMode);
        base_feedback = base_cyclic->RefreshFeedback();

        int actuator_count = base->GetActuatorCount().count();

        // Initialize each actuator to its current position
        for(int i = 0; i < actuator_count; i++)
        {
            commands.push_back(base_feedback.actuators(i).position());
            base_command.add_actuators()->set_position(base_feedback.actuators(i).position());
        }

        // Define the callback function used in Refresh_callback
        auto lambda_fct_callback = [](const Kinova::Api::Error &err, const k_api::BaseCyclic::Feedback data)
        {
            // We are printing the data of the moving actuator just for the example purpose,
            // avoid this in a real-time loop
            std::string serialized_data;
            google::protobuf::util::MessageToJsonString(data.actuators(data.actuators_size() - 1), &serialized_data);
            // std::cout << serialized_data << std::endl << std::endl;
        };

        lcm::LCM lcm;
        if(!lcm.good())
            return 1;

        Handler handlerObject;
        lcm.subscribe("KINOVA_COMMAND", &Handler::handleMessage, &handlerObject);

        // Open output file to save printed data with timestamp in its name
        time_t now_time = time(nullptr);
        struct tm* tm_info = localtime(&now_time);
        char time_buf[80];
        strftime(time_buf, sizeof(time_buf), "_%Y%m%d_%H%M%S", tm_info);
        std::string output_filename = std::string("/home/icl-baby/kinova_lcm/output") + time_buf + ".txt";
        std::ofstream output_file(output_filename, std::ios::out);

        while(0 == lcm.handle())
        {
            base_feedback = base_cyclic->RefreshFeedback();
            for (int i = 0; i < 7; i++)
            {
                joint_fbk_pos[i] = base_feedback.actuators(i).position();
                if (joint_fbk_pos[i] > 180)
                {
                    joint_fbk_pos[i] = joint_fbk_pos[i] - 360;
                }
                joint_fbk_pos[i] = joint_fbk_pos[i] * M_PI / 180;
                joint_fbk_spd[i] = base_feedback.actuators(i).velocity() * M_PI /180;
                joint_fbk_trq[i] = base_feedback.actuators(i).torque();
            }
            printf("joint_fbk_pos: %f %f %f %f %f %f %f\n", joint_fbk_pos[0], joint_fbk_pos[1], joint_fbk_pos[2], joint_fbk_pos[3], joint_fbk_pos[4], joint_fbk_pos[5], joint_fbk_pos[6]);
            printf("joint_fbk_spd: %f %f %f %f %f %f %f\n", joint_fbk_spd[0], joint_fbk_spd[1], joint_fbk_spd[2], joint_fbk_spd[3], joint_fbk_spd[4], joint_fbk_spd[5], joint_fbk_spd[6]);
            printf("joint_fbk_trq: %f %f %f %f %f %f %f\n", joint_fbk_trq[0], joint_fbk_trq[1], joint_fbk_trq[2], joint_fbk_trq[3], joint_fbk_trq[4], joint_fbk_trq[5], joint_fbk_trq[6]);
            // Save the same output to a txt file
            output_file << joint_fbk_trq[0] << " " << joint_fbk_trq[1] << " " << joint_fbk_trq[2] << " " 
                        << joint_fbk_trq[3] << " " << joint_fbk_trq[4] << " " << joint_fbk_trq[5] << " " << joint_fbk_trq[6] << " "
                        << joint_fbk_pos[0] << " " << joint_fbk_pos[1] << " " << joint_fbk_pos[2] << " " 
                        << joint_fbk_pos[3] << " " << joint_fbk_pos[4] << " " << joint_fbk_pos[5] << " " << joint_fbk_pos[6] << " "
                        << joint_fbk_spd[0] << " " << joint_fbk_spd[1] << " " << joint_fbk_spd[2] << " " 
                        << joint_fbk_spd[3] << " " << joint_fbk_spd[4] << " " << joint_fbk_spd[5] << " " << joint_fbk_spd[6] << "\n";
            for(int i = 0; i < actuator_count; i++)
            {
                base_command.mutable_actuators(i)->set_position(fmod(g_joint_pos_des[i] * 180 / M_PI, 360.0f));
            }
            try
            {
                base_cyclic->Refresh_callback(base_command, lambda_fct_callback, 0);
            }
            catch(...){} 
        }
        // Close the output file
        output_file.close();
    }
    catch (k_api::KDetailedException& ex)
    {
        std::cout << "Kortex error: " << ex.what() << std::endl;
        return_status = false;
    }
    catch (std::runtime_error& ex2)
    {
        std::cout << "Runtime error: " << ex2.what() << std::endl;
        return_status = false;
    }
 
    // Set back the servoing mode to Single Level Servoing
    servoingMode.set_servoing_mode(k_api::Base::ServoingMode::SINGLE_LEVEL_SERVOING);
    base->SetServoingMode(servoingMode);

    // Wait for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    return return_status;
}

bool example_angular_action_movement(k_api::Base::BaseClient* base) 
{
    std::cout << "Starting angular action movement ..." << std::endl;

    auto action = k_api::Base::Action();
    action.set_name("Example angular action movement");
    action.set_application_data("");

    auto reach_joint_angles = action.mutable_reach_joint_angles();
    auto joint_angles = reach_joint_angles->mutable_joint_angles();

    auto actuator_count = base->GetActuatorCount();

    // Arm straight up
    for (size_t i = 0; i < actuator_count.count(); ++i) 
    {
        auto joint_angle = joint_angles->add_joint_angles();
        joint_angle->set_joint_identifier(i);
        joint_angle->set_value(0);
    }

    std::promise<k_api::Base::ActionEvent> finish_promise;
    auto finish_future = finish_promise.get_future();
    auto promise_notification_handle = base->OnNotificationActionTopic(
        create_event_listener_by_promise(finish_promise),
        k_api::Common::NotificationOptions()
    );

    std::cout << "Executing action" << std::endl;
    base->ExecuteAction(action);

    std::cout << "Waiting for movement to finish ..." << std::endl;

    const auto status = finish_future.wait_for(TIMEOUT_DURATION);
    base->Unsubscribe(promise_notification_handle);

    if(status != std::future_status::ready)
    {
        std::cout << "Timeout on action notification wait" << std::endl;
        return false;
    }
    const auto promise_event = finish_future.get();

    std::cout << "Angular movement completed" << std::endl;
    std::cout << "Promise value : " << k_api::Base::ActionEvent_Name(promise_event) << std::endl; 

    return true;
}

int main(int argc, char **argv)
{
    auto parsed_args = ParseExampleArguments(argc, argv);

    // Create API objects
    auto error_callback = [](k_api::KError err){ cout << "_________ callback error _________" << err.toString(); };
    
    auto transport = new k_api::TransportClientTcp();
    auto router = new k_api::RouterClient(transport, error_callback);
    transport->connect(parsed_args.ip_address, PORT);

    auto transport_real_time = new k_api::TransportClientUdp();
    auto router_real_time = new k_api::RouterClient(transport_real_time, error_callback);
    transport_real_time->connect(parsed_args.ip_address, PORT_REAL_TIME);

    // Set session data connection information
    auto create_session_info = k_api::Session::CreateSessionInfo();
    create_session_info.set_username(parsed_args.username);
    create_session_info.set_password(parsed_args.password);
    create_session_info.set_session_inactivity_timeout(60000);   // (milliseconds)
    create_session_info.set_connection_inactivity_timeout(2000); // (milliseconds)

    // Session manager service wrapper
    std::cout << "Creating sessions for communication" << std::endl;
    auto session_manager = new k_api::SessionManager(router);
    session_manager->CreateSession(create_session_info);
    auto session_manager_real_time = new k_api::SessionManager(router_real_time);
    session_manager_real_time->CreateSession(create_session_info);
    std::cout << "Sessions created" << std::endl;

    // Create services
    auto base = new k_api::Base::BaseClient(router);
    auto base_cyclic = new k_api::BaseCyclic::BaseCyclicClient(router_real_time);


    // example_move_to_home_position(base);

    // std::cout << "home finished" << std::endl;

    example_angular_action_movement(base);

    std::cout << "000 finished" << std::endl;
    // // Example core
    // auto isOk = example_actuator_low_level_velocity_control(base, base_cyclic);
    // if (!isOk)
    // {
    //     std::cout << "There has been an unexpected error in example_cyclic_armbase() function." << std::endl;
    // }

    // Close API session

    LCM_test(base, base_cyclic);

    session_manager->CloseSession();
    session_manager_real_time->CloseSession();

    // Deactivate the router and cleanly disconnect from the transport object
    router->SetActivationStatus(false);
    transport->disconnect();
    router_real_time->SetActivationStatus(false);
    transport_real_time->disconnect();

    // Destroy the API
    delete base;
    delete base_cyclic;
    delete session_manager;
    delete session_manager_real_time;
    delete router;
    delete router_real_time;
    delete transport;
    delete transport_real_time;
}
