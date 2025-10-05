#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/collision/collision.hpp"

#include <iostream>

int main(int /*argc*/, char ** /*argv*/) {
    using namespace pinocchio;
    const std::string robots_model_path = "../rnd_torus/ur5e/";

    const std::string urdf_filename =
        std::string("../rnd_torus/ur5e/"
                    "ur5e_extract_calibrated_explicit_path.urdf");
    const std::string srdf_filename = std::string(
        "../rnd_torus/ur5e/ur5e_extract_calibrated_fixed.srdf");

    // Shelf URDF path
    const std::string shelf_urdf_filename =
        std::string("../rnd_torus/ur5e/shelf.urdf");

    // Load UR5e robot model
    Model robot_model;
    pinocchio::urdf::buildModel(urdf_filename, robot_model);
    Data robot_data(robot_model);

    // Load UR5e collision geometry
    GeometryModel robot_geom_model;
    pinocchio::urdf::buildGeom(
        robot_model, urdf_filename, pinocchio::COLLISION, robot_geom_model);

    // Load shelf model
    Model shelf_model;
    pinocchio::urdf::buildModel(shelf_urdf_filename, shelf_model);
    Data shelf_data(shelf_model);

    // Load shelf collision geometry
    GeometryModel shelf_geom_model;
    pinocchio::urdf::buildGeom(
        shelf_model, shelf_urdf_filename, pinocchio::COLLISION, shelf_geom_model);

    // Create combined geometry model for robot-only collisions
    GeometryModel combined_geom_model = robot_geom_model;

    // Add shelf geometries to combined model
    for (size_t i = 0; i < shelf_geom_model.geometryObjects.size(); ++i) {
        GeometryObject shelf_geom = shelf_geom_model.geometryObjects[i];

        // Set a fixed placement for the shelf (you can modify this position)
        SE3 shelf_placement = SE3::Identity();
        shelf_placement.translation() << 0.0, 0.75, 0.0;
        shelf_placement.rotation() = Eigen::Quaterniond(0.0, 0.0, 0.0, 1.0);
        shelf_geom.placement = shelf_placement * shelf_geom.placement;

        // Add to combined model
        combined_geom_model.addGeometryObject(shelf_geom);
    }

    // Add collision pairs: robot self-collisions
    combined_geom_model.addAllCollisionPairs();

    // Remove robot self-collision pairs based on SRDF
    pinocchio::srdf::removeCollisionPairs(
        robot_model, combined_geom_model, srdf_filename);

    // Add robot-shelf collision pairs
    size_t robot_geom_count = robot_geom_model.geometryObjects.size();
    size_t shelf_geom_count = shelf_geom_model.geometryObjects.size();

    for (size_t i = 0; i < robot_geom_count; ++i) {
        for (size_t j = 0; j < shelf_geom_count; ++j) {
            size_t shelf_geom_id = robot_geom_count + j;
            combined_geom_model.addCollisionPair(CollisionPair(i, shelf_geom_id));
        }
    }

    // Build the data associated to the combined geometry model
    GeometryData combined_geom_data(combined_geom_model);

    // Print geometry model information
    std::cout << "=== ROBOT GEOMETRIES ===" << std::endl;
    for (size_t i = 0; i < robot_geom_count; ++i) {
        std::cout << "Robot Geometry " << i << ": "
                  << combined_geom_model.geometryObjects[i].name << std::endl;
    }
    std::cout << "\n=== SHELF GEOMETRIES ===" << std::endl;
    for (size_t i = robot_geom_count;
         i < combined_geom_model.geometryObjects.size();
         ++i) {
        std::cout << "Shelf Geometry " << i << ": "
                  << combined_geom_model.geometryObjects[i].name << std::endl;
    }
    std::cout << "\nTotal geometries: " << combined_geom_model.ngeoms << std::endl;
    std::cout << "Total collision pairs: "
              << combined_geom_model.collisionPairs.size() << std::endl;
    std::cout << std::endl;

    Eigen::VectorXd q = Eigen::VectorXd::Zero(robot_model.nq);
    for (Model::JointIndex j = 0; j < robot_model.njoints; ++j) {
        std::cout << "Joint " << j << " : " << robot_model.names[j]
                  << " has index_qs " << robot_model.idx_qs[j] << " and nqs "
                  << robot_model.nqs[j] << std::endl;
    }

    // Test collision detection
    computeCollisions(
        robot_model, robot_data, combined_geom_model, combined_geom_data, q);

    // Print the status of all collision pairs
    std::cout << "\n=== COLLISION RESULTS ===" << std::endl;
    for (size_t k = 0; k < combined_geom_model.collisionPairs.size(); ++k) {
        const CollisionPair &cp = combined_geom_model.collisionPairs[k];
        const hpp::fcl::CollisionResult &cr =
            combined_geom_data.collisionResults[k];

        std::string geom1_name =
            combined_geom_model.geometryObjects[cp.first].name;
        std::string geom2_name =
            combined_geom_model.geometryObjects[cp.second].name;

        // Identify if this is robot-robot or robot-shelf collision
        bool is_robot_shelf =
            (cp.first < robot_geom_count && cp.second >= robot_geom_count) ||
            (cp.first >= robot_geom_count && cp.second < robot_geom_count);

        std::string pair_type = is_robot_shelf ? "[ROBOT-SHELF]" : "[ROBOT-ROBOT]";

        std::cout << pair_type << " collision pair: " << cp.first << "("
                  << geom1_name << ") , " << cp.second << "(" << geom2_name
                  << ") - collision: ";
        std::cout << (cr.isCollision() ? "YES" : "no") << std::endl;
    }

    // stop at first collision
    computeCollisions(
        robot_model, robot_data, combined_geom_model, combined_geom_data, q, true);

    // is there a collision?
    std::cout << "Is there a collision? "
              << (combined_geom_data
                          .collisionResults[combined_geom_data.collisionPairIndex]
                          .isCollision()
                      ? "YES"
                      : "no")
              << std::endl;
    return 0;
};