/*
State space: R2
Limit joints : [-pi, pi] strictly
Robot: PlanarRR
Environment: Rectangles
Given:
    - qstart : explicitly
    - qgoal : explicitly
Problem : Find a path from qstart to qgoal as usual.
Planners:
    - RRT
    - RRTConnect
    - RRTstar
*/
#include "sim_planar_rr.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/goals/GoalState.h>
#include <ompl-1.5/ompl/base/spaces/SO2StateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRT.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTstar.h>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state, PlanarRRSIM &sim);
void savePathToFile(const og::PathGeometric &path, const std::string &filename);
void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename);
void saveStartAndGoal(const ompl::geometric::SimpleSetup &ss,
                      const std::string &filename);

int main() {
    // read data from YAML
    YAML::Node config = YAML::LoadFile("../config/so2s_snggoal.yaml");
    double l1 = config["robot"]["l1"].as<double>();
    double l2 = config["robot"]["l2"].as<double>();
    auto rectangles = config["env"]["rectangles"];
    std::vector<double> qstart = config["qstart"].as<std::vector<double>>();
    std::vector<double> qgoal = config["qgoal"].as<std::vector<double>>();
    double range = config["range"].as<double>();
    double bias = config["bias"].as<double>();
    double time_limit = config["time_limit"].as<double>();
    bool simplify_solution = config["simplify_solution"].as<bool>();
    const char *varrsrc = std::getenv("RSRC_DIR");
    std::string save_planner_graphml =
        std::string(varrsrc) + "/rnd_torus/" +
        config["save_planner_data"].as<std::string>() + ".graphml";
    std::string save_start_goal = std::string(varrsrc) + "/rnd_torus/" +
                                  config["save_start_goal"].as<std::string>() +
                                  ".csv";
    std::string save_path = std::string(varrsrc) + "/rnd_torus/" +
                            config["save_path"].as<std::string>() + ".csv";

    // Robot setup
    PlanarRR robot(l1, l2);

    // Simulation setup
    std::vector<Rectangle> env;
    for (const auto &rect : rectangles) {
        env.push_back(Rectangle(rect[0].as<double>(),
                                rect[1].as<double>(),
                                rect[2].as<double>(),
                                rect[3].as<double>()));
    }
    PlanarRRSIM sim(robot, env);

    // Planning space setup
    auto space = std::make_shared<ob::CompoundStateSpace>(); // each joint
                                                             // represented by SO2
    // First joint with weight of 1.0
    space->addSubspace(std::make_shared<ob::SO2StateSpace>(), 1.0);
    // Second joint
    space->addSubspace(std::make_shared<ob::SO2StateSpace>(), 1.0);
    og::SimpleSetup ss(space);

    // Collision setup. Set the state validity checker using a lambda
    ss.setStateValidityChecker(
        [&sim](const ob::State *state) { return isStateValid(state, sim); });

    // start and goal states
    ob::ScopedState<ob::CompoundStateSpace> start(space);
    start[0] = qstart[0];
    start[1] = qstart[1];
    ob::ScopedState<ob::CompoundStateSpace> goal(space);
    goal[0] = qgoal[1];
    goal[1] = qgoal[1];
    ss.setStartAndGoalStates(start, goal);

    // Planner setup and solved
    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    planner->setRange(range);
    planner->setGoalBias(bias);
    ss.setPlanner(planner);
    ob::PlannerStatus solved = ss.solve(time_limit);

    if (solved) {
        std::cout << "Found solution! Printing it out...!" << std::endl;

        if (simplify_solution) {
            ss.simplifySolution();
        }
        ss.getSolutionPath().print(std::cout);
        ss.getSolutionPath().printAsMatrix(std::cout);

        const og::PathGeometric &path = ss.getSolutionPath();
        auto space_information(
            std::make_shared<ompl::base::SpaceInformation>(space));
        savePathToFile(path, save_path);
        savePlannerData(planner, space_information, save_planner_graphml);
        saveStartAndGoal(ss, save_start_goal);
    } else {
        std::cout << "No solution found." << std::endl;
    }

    return 0;
}

bool isStateValid(const ob::State *state, PlanarRRSIM &sim) {
    const auto *compoundstate = state->as<ob::CompoundState>();
    double theta1 = compoundstate->as<ob::SO2StateSpace::StateType>(0)->value;
    double theta2 = compoundstate->as<ob::SO2StateSpace::StateType>(1)->value;
    bool c = sim.check_collision(theta1, theta2);
    // printf("state {%f, %f} - col {%d} \n", theta1, theta2, c);
    return !c;
}

void savePathToFile(const og::PathGeometric &path, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        // Extract joint angles (SO2 states)
        const ob::State *state = path.getState(i);
        const auto *compoundState = state->as<ob::CompoundState>();
        double joint1 = compoundState->as<ob::SO2StateSpace::StateType>(0)->value;
        double joint2 = compoundState->as<ob::SO2StateSpace::StateType>(1)->value;
        file << joint1 << "," << joint2 << std::endl;
    }

    file.close();
    std::cout << "Path saved to " << filename << std::endl;
}

void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename) {
    ompl::base::PlannerData plannerData(si);
    planner->getPlannerData(plannerData);
    std::ofstream file(filename);
    plannerData.printGraphML(file);
    file.close();
}

void saveStartAndGoal(const ompl::geometric::SimpleSetup &ss,
                      const std::string &filename) {
    // Get the start and goal states
    const ob::State *startState = ss.getProblemDefinition()->getStartState(0);

    std::ofstream file(filename);

    const auto *start_coords = startState->as<ob::CompoundState>();
    double ss1 = start_coords->as<ob::SO2StateSpace::StateType>(0)->value;
    double ss2 = start_coords->as<ob::SO2StateSpace::StateType>(1)->value;
    file << ss1 << "," << ss2 << std::endl;

    auto goalPtr = ss.getProblemDefinition()->getGoal();
    auto goalState = std::dynamic_pointer_cast<ob::GoalState>(goalPtr);
    const ob::State *goalStateVal = goalState->getState();
    const auto *compoundState = goalStateVal->as<ob::CompoundState>();
    double gs1 = compoundState->as<ob::SO2StateSpace::StateType>(0)->value;
    double gs2 = compoundState->as<ob::SO2StateSpace::StateType>(1)->value;
    file << gs1 << "," << gs2 << std::endl;

    file.close();
}