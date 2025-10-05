/*
State space: R2
Limit joints: [-2pi, 2pi] strictly
Robot: Planar6R
Environment: Rectangles
6iven:
    - qstart : explicitly
    - qgoal : explicitly and then find alternative goal configurations as set a
goal set.
Problem : Find a path from qstart to qgoal as usual.
Planners:
    - RRT
    - RRTstar
    - RRTConnect
*/
#include "findaltconfig.h"
#include "sim_planar_6r.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/goals/GoalStates.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRT.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTstar.h>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state, Planar6RSIM &sim);
void savePathToFile(const og::PathGeometric &path, const std::string &filename);
void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename);
void saveStartAndGoal(const ompl::geometric::SimpleSetup &ss,
                      const std::string &filename);

int main() {
    // Load YAML configuration
    YAML::Node config = YAML::LoadFile("../config/r6s_altgoal.yaml");
    double l1 = config["robot"]["l1"].as<double>();
    double l2 = config["robot"]["l2"].as<double>();
    double l3 = config["robot"]["l3"].as<double>();
    double l4 = config["robot"]["l4"].as<double>();
    double l5 = config["robot"]["l5"].as<double>();
    double l6 = config["robot"]["l6"].as<double>();
    auto rectangles = config["env"]["rectangles"];
    std::vector<double> qlimit = config["qlimit"].as<std::vector<double>>();
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
    Planar6R robot(l1, l2, l3, l4, l5, l6);

    // Simulation environment setup
    std::vector<Rectangle> env;
    for (const auto &rect : rectangles) {
        env.push_back(Rectangle(rect[0].as<double>(),
                                rect[1].as<double>(),
                                rect[2].as<double>(),
                                rect[3].as<double>()));
    }
    // Robot setup
    Planar6RSIM sim(robot, env);

    // Planning space setup
    auto space = std::make_shared<ob::RealVectorStateSpace>(6);
    ob::RealVectorBounds bounds(6);
    for (size_t i = 0; i < 6; i++) {
        bounds.setLow(i, -qlimit[i]);
        bounds.setHigh(i, qlimit[i]);
    }
    space->setBounds(bounds);
    og::SimpleSetup ss(space);

    // Collision setup
    ss.setStateValidityChecker(
        [&sim](const ob::State *state) { return isStateValid(state, sim); });

    // Define the start state
    ob::ScopedState<> start(space);
    for (size_t i = 0; i < 6; ++i) {
        start[i] = qstart[i];
    }
    ss.setStartState(start);

    // Define multiple goal states
    Eigen::RowVectorXd qgoal_eig(6);
    for (size_t i = 0; i < 6; i++) {
        qgoal_eig[i] = qgoal[i];
    }

    Eigen::RowVectorXd qlimit_eig(6);
    for (size_t i = 0; i < 6; i++) {
        qlimit_eig[i] = qlimit[i];
    }

    Eigen::MatrixXd qgoalalt = find_alt_config(qgoal_eig, qlimit_eig);
    printf("\nI found %d alternative goal configurations.\n", qgoalalt.rows());
    printf("\nThey are: \n");
    for (size_t i = 0; i < qgoalalt.rows(); i++) {
        std::cout << qgoalalt.row(i) << std::endl;
    }
    auto goal = std::make_shared<ob::GoalStates>(ss.getSpaceInformation());
    for (size_t i = 0; i < qgoalalt.rows(); i++) {
        ob::ScopedState<> goali(space);
        for (size_t j = 0; j < qgoalalt.cols(); j++) {
            goali[j] = qgoalalt.row(i)[j];
        }
        goal->addState(goali);
    }
    ss.setGoal(goal);

    // Planner setup and solve
    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    planner->setRange(range);
    planner->setGoalBias(bias);
    ss.setPlanner(planner);
    ob::PlannerStatus solved = ss.solve(time_limit);

    if (solved) {
        std::cout << "Found solution:" << std::endl;

        if (simplify_solution) {
            ss.simplifySolution();
        }

        ss.getSolutionPath().print(std::cout);

        const og::PathGeometric &path = ss.getSolutionPath();
        auto space_information(
            std::make_shared<ompl::base::SpaceInformation>(space));
        savePathToFile(path, save_path);
        savePlannerData(planner, space_information, save_planner_graphml);
        saveStartAndGoal(ss, save_start_goal);
    } else {
        std::cout << "No solution found" << std::endl;
    }

    return 0;
}

bool isStateValid(const ob::State *state, Planar6RSIM &sim) {
    const ob::RealVectorStateSpace::StateType *realState =
        state->as<ob::RealVectorStateSpace::StateType>();
    double theta1 = realState->values[0];
    double theta2 = realState->values[1];
    double theta3 = realState->values[2];
    double theta4 = realState->values[3];
    double theta5 = realState->values[4];
    double theta6 = realState->values[5];
    bool c = sim.check_collision(theta1, theta2, theta3, theta4, theta5, theta6);
    return !c;
}

void savePathToFile(const og::PathGeometric &path, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const ob::State *state = path.getState(i);
        const auto *realState = state->as<ob::RealVectorStateSpace::StateType>();
        double theta1 = realState->values[0];
        double theta2 = realState->values[1];
        double theta3 = realState->values[2];
        double theta4 = realState->values[3];
        double theta5 = realState->values[4];
        double theta6 = realState->values[5];
        file << theta1 << "," << theta2 << std::endl;
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
    const ob::GoalStates *goalState =
        ss.getProblemDefinition()->getGoal()->as<ob::GoalStates>();

    std::ofstream file(filename);

    const double *start_coords =
        startState->as<ob::RealVectorStateSpace::StateType>()->values;
    file << start_coords[0] << "," << start_coords[1] << "," << start_coords[2]
         << "," << start_coords[3] << "," << start_coords[4] << ","
         << start_coords[5] << std::endl;

    // goals is second row onward
    int numGoals = goalState->getStateCount();
    printf("\nThere are %d goalstate(s).\n", numGoals);
    for (size_t i = 0; i < numGoals; i++) {
        const double *goal_coords = goalState->getState(i)
                                        ->as<ob::RealVectorStateSpace::StateType>()
                                        ->values;
        file << goal_coords[0] << "," << goal_coords[1] << "," << goal_coords[2]
             << "," << goal_coords[3] << "," << goal_coords[4] << ","
             << goal_coords[5] << std::endl;
    }
    file.close();
}