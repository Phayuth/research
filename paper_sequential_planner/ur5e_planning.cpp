#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ompl-1.5/ompl/base/goals/GoalState.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRT.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTstar.h>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool readYAMLConfig(const std::string &filename, YAML::Node &config);
bool isStateValid(const ob::State *state, UR5eSIM &sim);
void savePathToFile(const og::PathGeometric &path, const std::string &filename);
void savePlannerData(const ompl::base::PlannerPtr &planner,
                     const ompl::base::SpaceInformationPtr &si,
                     const std::string &filename);
void saveStartAndGoal(const ompl::geometric::SimpleSetup &ss,
                      const std::string &filename);

int main(int argc, char const *argv[]) {
    return 0;
}
