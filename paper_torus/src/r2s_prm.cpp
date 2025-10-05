/*
State space: R2
Limit joints : [-pi, pi] or [-2pi, 2pi]
Robot: PlanarRR
Environment: Rectangles
Given:
    - qstart : explicitly
    - qgoal : explicitly
Problem : Build the PRM graph only. It's used to find path later.
Planners:
    - PRM
    - PRMstar
*/
#include "findaltconfig.h"
#include "sim_planar_rr.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/PlannerDataStorage.h>
#include <ompl-1.5/ompl/base/PlannerTerminationCondition.h>
#include <ompl-1.5/ompl/base/goals/GoalStates.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state, PlanarRRSIM &sim);
void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename);
void savePlannerStorage(ob::PlannerDataStorage &datastorage, ob::PlannerData &data,
                        std::string &filename);
void loadPlannerStoarge(ob::PlannerDataStorage &datastorage, ob::PlannerData &data,
                        std::string &filename);

int main() {
    // Load YAML
    YAML::Node config = YAML::LoadFile("../config/r2s_prm.yaml");
    double l1 = config["robot"]["l1"].as<double>();
    double l2 = config["robot"]["l2"].as<double>();
    auto rectangles = config["env"]["rectangles"];
    std::vector<double> qlimit = config["qlimit"].as<std::vector<double>>();
    std::vector<double> qstart = config["qstart"].as<std::vector<double>>();
    std::vector<double> qgoal = config["qgoal"].as<std::vector<double>>();
    double time_limit = config["time_limit"].as<double>();
    bool simplify_solution = config["simplify_solution"].as<bool>();
    const char *varrsrc = std::getenv("RSRC_DIR");
    std::string save_planner_graphml =
        std::string(varrsrc) + "/rnd_torus/" +
        config["save_planner_data"].as<std::string>() + ".graphml";
    std::string save_planner_bin =
        std::string(varrsrc) + "/rnd_torus/" +
        config["save_planner_data_bin"].as<std::string>();
    std::string save_start_goal = std::string(varrsrc) + "/rnd_torus/" +
                                  config["save_start_goal"].as<std::string>() +
                                  ".csv";
    std::string save_path = std::string(varrsrc) + "/rnd_torus/" +
                            config["save_path"].as<std::string>() + ".csv";

    // Robot setup
    PlanarRR robot(l1, l2);

    // Simulation environment setup
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
    for (int i = 0; i < 2; ++i) {
        bounds.setLow(i, -qlimit[i]);
        bounds.setHigh(i, qlimit[i]);
    }
    space->setBounds(bounds);
    og::SimpleSetup ss(space);

    // Collision setup
    ss.setStateValidityChecker(
        [&sim](const ob::State *state) { return isStateValid(state, sim); });

    // start and goal states
    ob::ScopedState<> start(space);
    start[0] = qstart[0];
    start[1] = qstart[1];
    ob::ScopedState<> goal(space);
    goal[0] = qgoal[0];
    goal[1] = qgoal[1];
    ss.setStartAndGoalStates(start, goal);

    // Planner setup and solve
    // auto planner = std::make_shared<og::PRM>(ss.getSpaceInformation());
    auto planner = std::make_shared<og::PRMstar>(ss.getSpaceInformation());
    ss.setPlanner(planner);
    ob::PlannerStatus solved = ss.solve(time_limit);

    // storage
    ob::PlannerDataStorage datastorage;

    ob::PlannerData data(ss.getSpaceInformation());
    ss.getPlannerData(data);

    // save to boost graph
    savePlannerData(planner, ss.getSpaceInformation(), save_planner_graphml);
    savePlannerStorage(datastorage, data, save_planner_bin);
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

void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename) {
    ompl::base::PlannerData plannerData(si);
    planner->getPlannerData(plannerData);
    std::ofstream file(filename);
    plannerData.printGraphML(file);
    file.close();
}

void savePlannerStorage(ob::PlannerDataStorage &datastorage, ob::PlannerData &data,
                        std::string &filename) {
    datastorage.store(data, filename.c_str());
}

void loadPlannerStoarge(ob::PlannerDataStorage &datastorage, ob::PlannerData &data,
                        std::string &filename) {
    datastorage.load(filename.c_str(), data);
}
