/*
State space: R2
Limit joint: [-pi, pi] or [-2pi, 2pi]
Robot: PlanarRR
Environment: Rectangles
Given:
    - qstart : explicitly
    - qgoal : implicitly (defined by a goal region in Cartesian space)
Problem : Find a path from qstart to a goal. The goal is near by defined a distance
         tolerance in Cartesian space.
Planners:
    - RRT
    - RRTConnect : it needs another explicit goal but we don't have it.
    - RRTstar
*/
#include "sim_planar_rr.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/goals/GoalState.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/base/spaces/SE2StateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRT.h>
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

class CartesianGoalRegion : public ob::GoalRegion {
    public:
        CartesianGoalRegion(const ob::SpaceInformationPtr &si, PlanarRR &robot,
                            double x_goal, double y_goal, double tol)
            : ob::GoalRegion(si), robot_(robot), x_goal_(x_goal), y_goal_(y_goal),
              tol_(tol) {
            threshold_ = tol;
        }

        double distanceGoal(const ob::State *state) const override {
            const auto *realState =
                state->as<ob::RealVectorStateSpace::StateType>();
            double theta1 = realState->values[0];
            double theta2 = realState->values[1];
            std::array<double, 2> xy;
            xy = robot_.forward_kinematic(theta1, theta2);
            double dx = xy[0] - x_goal_;
            double dy = xy[1] - y_goal_;
            return std::sqrt(dx * dx + dy * dy);
        }

    public:
        PlanarRR &robot_;
        double x_goal_, y_goal_, tol_;
};

int main() {
    // read YAML configurations
    YAML::Node config = YAML::LoadFile("../config/r2s_cartgoal.yaml");
    double l1 = config["robot"]["l1"].as<double>();
    double l2 = config["robot"]["l2"].as<double>();
    auto rectangles = config["env"]["rectangles"];
    std::vector<double> qlimit = config["qlimit"].as<std::vector<double>>();
    std::vector<double> qstart = config["qstart"].as<std::vector<double>>();
    double xgoal = config["xgoal"].as<double>();
    double ygoal = config["ygoal"].as<double>();
    double goal_tolerance = config["goal_tolerance"].as<double>();
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
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds bounds(2);
    for (size_t i = 0; i < 2; i++) {
        bounds.setLow(i, -qlimit[i]);
        bounds.setHigh(i, qlimit[i]);
    }
    space->setBounds(bounds);
    og::SimpleSetup ss(space);

    ss.setStateValidityChecker(
        [&sim](const ob::State *state) { return isStateValid(state, sim); });

    // start and goal states
    ob::ScopedState<> start(space);
    start[0] = qstart[0];
    start[1] = qstart[1];
    ss.setStartState(start);

    // Instead of ob::ScopedState goal...
    auto goal_region = std::make_shared<CartesianGoalRegion>(
        ss.getSpaceInformation(), robot, xgoal, ygoal, goal_tolerance);
    ss.setGoal(goal_region);

    // Planner setup and solved
    // auto planner = std::make_shared<og::RRT>(ss.getSpaceInformation());
    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    planner->setRange(range);
    planner->setGoalBias(bias);
    ss.setPlanner(planner);
    ob::PlannerStatus solved = ss.solve(time_limit);

    if (solved) {
        std::cout << "Found solution!" << std::endl;

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
        std::cout << "No solution found." << std::endl;
    }

    return 0;
}

bool isStateValid(const ob::State *state, PlanarRRSIM &sim) {
    const ob::RealVectorStateSpace::StateType *realState =
        state->as<ob::RealVectorStateSpace::StateType>();
    double theta1 = realState->values[0];
    double theta2 = realState->values[1];
    bool c = sim.check_collision(theta1, theta2);
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
    const CartesianGoalRegion *goalState =
        ss.getProblemDefinition()->getGoal()->as<CartesianGoalRegion>();

    std::ofstream file(filename);

    const double *start_coords =
        startState->as<ob::RealVectorStateSpace::StateType>()->values;
    file << start_coords[0] << "," << start_coords[1] << std::endl;

    file << goalState->x_goal_ << "," << goalState->y_goal_ << std::endl;

    file.close();
}