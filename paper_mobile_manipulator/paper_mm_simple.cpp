/*
A simple OMPL path planning for mobile manipulator with 9DOFs using RRT*.
*/
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/goals/GoalState.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/base/spaces/SO2StateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTstar.h>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state);
void savePathToFile(const og::PathGeometric &path, const std::string &filename);

int main() {
    YAML::Node config = YAML::LoadFile("../config/simple_plan.yaml");

    auto space = std::make_shared<ob::CompoundStateSpace>();

    // xy
    auto s1 = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds sb1(2);
    for (size_t i = 0; i < 2; i++) {
        printf("low : %f and high : %f \n",
               -config["qlimit"][i].as<double>(),
               config["qlimit"][i].as<double>());
        sb1.setLow(i, -config["qlimit"][i].as<double>());
        sb1.setHigh(i, config["qlimit"][i].as<double>());
    }
    s1->setBounds(sb1);
    space->addSubspace(s1, 1.0);

    // yaw
    space->addSubspace(std::make_shared<ob::SO2StateSpace>(), 1.0);

    // 6 joints arm
    auto s3 = std::make_shared<ob::RealVectorStateSpace>(6);
    ob::RealVectorBounds sb3(6);
    for (size_t i = 0; i < 6; i++) {
        printf("low : %f and high : %f \n",
               -config["qlimit"][i + 3].as<double>(),
               config["qlimit"][i + 3].as<double>());
        sb3.setLow(i, -config["qlimit"][i + 3].as<double>());
        sb3.setHigh(i, config["qlimit"][i + 3].as<double>());
    }
    s3->setBounds(sb3);
    space->addSubspace(s3, 1.0);
    og::SimpleSetup ss(space);

    printf("There are %d configuations.\n", space->getDimension());

    ss.setStateValidityChecker(isStateValid);

    ob::ScopedState<ob::CompoundStateSpace> start(space);
    ob::ScopedState<ob::CompoundStateSpace> goal(space);
    for (unsigned int i = 0; i < space->getDimension(); i++) {
        start[i] = config["qstart"][i].as<double>();
        goal[i] = config["qgoal"][i].as<double>();
    }
    ss.setStartAndGoalStates(start, goal);

    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    planner->setRange(config["range"].as<double>());
    planner->setGoalBias(config["bias"].as<double>());
    ss.setPlanner(planner);
    ob::PlannerStatus solved = ss.solve(config["time_limit"].as<double>());

    if (solved) {
        std::cout << "Found solutions " << std::endl;

        if (config["simplify_solution"].as<bool>()) {
            ss.simplifySolution();
        }
        ss.getSolutionPath().print(std::cout);
        ss.getSolutionPath().printAsMatrix(std::cout);

        const og::PathGeometric &path = ss.getSolutionPath();
        savePathToFile(path, config["save_path"].as<std::string>() + ".csv");
    } else {
        std::cout << "No solution found." << std::endl;
    }
    return 0;
}

bool isStateValid(const ob::State *state) {
    return true;
}

void savePathToFile(const og::PathGeometric &path, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < path.getStateCount(); i++) {
        const ob::State *state = path.getState(i);
        const auto *cpstate = state->as<ob::CompoundState>();
        double q0 = cpstate->as<ob::RealVectorStateSpace::StateType>(0)->values[0];
        double q1 = cpstate->as<ob::RealVectorStateSpace::StateType>(0)->values[1];
        double q2 = cpstate->as<ob::SO2StateSpace::StateType>(1)->value;
        double q3 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[0];
        double q4 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[1];
        double q5 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[2];
        double q6 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[3];
        double q7 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[4];
        double q8 = cpstate->as<ob::RealVectorStateSpace::StateType>(2)->values[5];
        file << q0 << "," << q1 << "," << q2 << "," << q3 << "," << q4 << "," << q5
             << "," << q6 << "," << q7 << "," << q8 << std::endl;
    }
    file.close();
    std::cout << "Path saved to " << filename << std::endl;
}