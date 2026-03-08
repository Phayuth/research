#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ompl-1.5/ompl/base/goals/GoalState.h>
#include <ompl-1.5/ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl-1.5/ompl/geometric/SimpleSetup.h>
#include <ompl-1.5/ompl/geometric/planners/rrt/RRTConnect.h>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

class OMPLPlanner {
    public:
        OMPLPlanner(int dim, const std::vector<double> &qlimit, double range,
                    double bias, double time_limit, bool simplify_solution);

        void setStartAndGoal(const std::vector<double> &qstart,
                             const std::vector<double> &qgoal);

        void
        setStateValidityChecker(const std::function<bool(const ob::State *)> &fn) {
            ss_.setStateValidityChecker(fn);
        }

        bool solve();

        const og::PathGeometric &getSolutionPath() const {
            return ss_.getSolutionPath();
        }

    private:
        ob::StateSpacePtr space_;
        og::SimpleSetup ss_;
        ompl::base::PlannerPtr planner_;
        double range_;
        double bias_;
        double time_limit_;
        bool simplify_solution_;
};

OMPLPlanner::OMPLPlanner(int dim, const std::vector<double> &qlimit, double range,
                         double bias, double time_limit, bool simplify_solution)
    : space_(std::make_shared<ob::RealVectorStateSpace>(dim)), ss_(space_),
      range_(range), bias_(bias), time_limit_(time_limit),
      simplify_solution_(simplify_solution) {

    // set bounds
    ob::RealVectorBounds bounds(dim);
    for (int i = 0; i < dim; ++i) {
        bounds.setLow(i, -qlimit[i]);
        bounds.setHigh(i, qlimit[i]);
    }
    space_->as<ob::RealVectorStateSpace>()->setBounds(bounds);

    // Default validity checker: accepts all states (replace externally if
    // needed).
    ss_.setStateValidityChecker([](const ob::State *) { return true; });

    // default planner: RRTConnect
    planner_ = std::make_shared<og::RRTConnect>(ss_.getSpaceInformation());
    if (auto r = std::dynamic_pointer_cast<og::RRTConnect>(planner_)) {
        r->setRange(range_);
    }
    ss_.setPlanner(planner_);
}

void OMPLPlanner::setStartAndGoal(const std::vector<double> &qstart,
                                  const std::vector<double> &qgoal) {
    ob::ScopedState<> start(space_);
    ob::ScopedState<> goal(space_);
    for (std::size_t i = 0; i < qstart.size() && i < 6; ++i)
        start[i] = qstart[i];
    for (std::size_t i = 0; i < qgoal.size() && i < 6; ++i)
        goal[i] = qgoal[i];
    ss_.setStartAndGoalStates(start, goal);
}

bool OMPLPlanner::solve() {
    ob::PlannerStatus solved = ss_.solve(time_limit_);

    if (solved) {
        std::cout << "Found solution!" << std::endl;
        if (simplify_solution_)
            ss_.simplifySolution();
        ss_.getSolutionPath().print(std::cout);
        return true;
    }
    std::cout << "No solution found." << std::endl;
    return false;
}

int main() {
    // read YAML configurations
    YAML::Node config = YAML::LoadFile("../config/r6s_snggoal.yaml");
    std::vector<double> qlimit = config["qlimit"].as<std::vector<double>>();
    std::vector<double> qstart = config["qstart"].as<std::vector<double>>();
    std::vector<double> qgoal = config["qgoal"].as<std::vector<double>>();
    double range = config["range"].as<double>();
    double bias = config["bias"].as<double>();
    double time_limit = config["time_limit"].as<double>();
    bool simplify_solution = config["simplify_solution"].as<bool>();

    OMPLPlanner planner(6, qlimit, range, bias, time_limit, simplify_solution);

    planner.setStartAndGoal(qstart, qgoal);
    planner.solve();

    return 0;
}