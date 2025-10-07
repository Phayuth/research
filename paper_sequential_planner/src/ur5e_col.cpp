#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/collision/collision.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct CollisionObject {
        std::string name;
        std::string urdf_file;
        pinocchio::SE3 pose;
        std::vector<std::string> no_collision_links;
};

class UR5eCollisionScene {
    private:
        pinocchio::Model robot_model_;
        pinocchio::Data robot_data_;
        pinocchio::GeometryModel robot_geom_model_;
        pinocchio::GeometryModel combined_geom_model_;
        pinocchio::GeometryData combined_geom_data_;

        std::string robot_urdf_file_;
        std::string robot_srdf_file_;
        std::vector<CollisionObject> collision_objects_;

    public:
        UR5eCollisionScene() = default;

        bool loadFromYAML(const std::string &yaml_file) {
            try {
                YAML::Node config = YAML::LoadFile(yaml_file);

                // Load robot configuration
                if (config["robot"]) {
                    robot_urdf_file_ = expandPackagePath(
                        config["robot"]["urdf_file"].as<std::string>());
                    robot_srdf_file_ = expandPackagePath(
                        config["robot"]["srdf_file"].as<std::string>());
                } else {
                    std::cerr << "Error: No robot configuration found in YAML file"
                              << std::endl;
                    return false;
                }

                // Load collision objects
                if (config["collision_urdf"]) {
                    for (const auto &obj_config : config["collision_urdf"]) {
                        CollisionObject obj;
                        obj.name = obj_config["name"].as<std::string>();
                        obj.urdf_file = expandPackagePath(
                            obj_config["urdf_file"].as<std::string>());

                        // Parse pose [x, y, z, qx, qy, qz, qw]
                        if (obj_config["pose"]) {
                            auto pose_vec =
                                obj_config["pose"].as<std::vector<double>>();
                            if (pose_vec.size() == 7) {
                                Eigen::Vector3d translation(
                                    pose_vec[0], pose_vec[1], pose_vec[2]);
                                Eigen::Quaterniond quaternion(
                                    pose_vec[6],
                                    pose_vec[3],
                                    pose_vec[4],
                                    pose_vec[5]); // w, x, y, z
                                obj.pose = pinocchio::SE3(
                                    quaternion.toRotationMatrix(), translation);
                            } else {
                                obj.pose = pinocchio::SE3::Identity();
                            }
                        } else {
                            obj.pose = pinocchio::SE3::Identity();
                        }

                        // Parse no_collision_links if present
                        if (obj_config["no_collision_links"]) {
                            obj.no_collision_links =
                                obj_config["no_collision_links"]
                                    .as<std::vector<std::string>>();
                        }

                        collision_objects_.push_back(obj);
                    }
                }

                return true;
            } catch (const YAML::Exception &e) {
                std::cerr << "Error parsing YAML file: " << e.what() << std::endl;
                return false;
            }
        }

        bool buildCollisionScene() {
            try {
                // Load robot model
                pinocchio::urdf::buildModel(robot_urdf_file_, robot_model_);
                robot_data_ = pinocchio::Data(robot_model_);

                // Load robot collision geometry
                pinocchio::urdf::buildGeom(robot_model_,
                                           robot_urdf_file_,
                                           pinocchio::COLLISION,
                                           robot_geom_model_);

                // Create combined geometry model starting with robot
                combined_geom_model_ = robot_geom_model_;

                // Add collision objects
                for (const auto &collision_obj : collision_objects_) {
                    addCollisionObject(collision_obj);
                }

                // Setup collision pairs
                setupCollisionPairs();

                // Build geometry data
                combined_geom_data_ =
                    pinocchio::GeometryData(combined_geom_model_);

                return true;
            } catch (const std::exception &e) {
                std::cerr << "Error building collision scene: " << e.what()
                          << std::endl;
                return false;
            }
        }

        bool checkCollision(const Eigen::VectorXd &q) {
            return pinocchio::computeCollisions(robot_model_,
                                                robot_data_,
                                                combined_geom_model_,
                                                combined_geom_data_,
                                                q,
                                                true);
        }

        void printCollisionInfo() {
            size_t robot_geom_count = robot_geom_model_.geometryObjects.size();

            std::cout << "=== ROBOT GEOMETRIES ===" << std::endl;
            for (size_t i = 0; i < robot_geom_count; ++i) {
                std::cout << "Robot Geometry " << i << ": "
                          << combined_geom_model_.geometryObjects[i].name
                          << std::endl;
            }

            std::cout << "\n=== COLLISION OBJECT GEOMETRIES ===" << std::endl;
            for (size_t i = robot_geom_count; i < combined_geom_model_.ngeoms;
                 ++i) {
                std::cout << "Collision Geometry " << i << ": "
                          << combined_geom_model_.geometryObjects[i].name
                          << std::endl;
            }

            std::cout << "\nTotal geometries: " << combined_geom_model_.ngeoms
                      << std::endl;
            std::cout << "Total collision pairs: "
                      << combined_geom_model_.collisionPairs.size() << std::endl;
        }

        void printDetailedCollisionResults(const Eigen::VectorXd &q) {
            pinocchio::computeCollisions(robot_model_,
                                         robot_data_,
                                         combined_geom_model_,
                                         combined_geom_data_,
                                         q);

            size_t robot_geom_count = robot_geom_model_.geometryObjects.size();

            std::cout << "\n=== COLLISION RESULTS ===" << std::endl;
            for (size_t k = 0; k < combined_geom_model_.collisionPairs.size();
                 ++k) {
                const pinocchio::CollisionPair &cp =
                    combined_geom_model_.collisionPairs[k];

                std::string geom1_name =
                    combined_geom_model_.geometryObjects[cp.first].name;
                std::string geom2_name =
                    combined_geom_model_.geometryObjects[cp.second].name;

                bool is_robot_object =
                    (cp.first < robot_geom_count &&
                     cp.second >= robot_geom_count) ||
                    (cp.first >= robot_geom_count && cp.second < robot_geom_count);

                std::string pair_type =
                    is_robot_object ? "[ROBOT-OBJECT]" : "[ROBOT-ROBOT]";

                bool collision = pinocchio::computeCollision(
                    combined_geom_model_, combined_geom_data_, k);

                std::cout << pair_type << " collision pair: " << cp.first << "("
                          << geom1_name << ") , " << cp.second << "(" << geom2_name
                          << ") - collision: " << (collision ? "YES" : "no")
                          << std::endl;
            }
        }

        const pinocchio::Model &getRobotModel() const {
            return robot_model_;
        }
        const pinocchio::GeometryModel &getCombinedGeometryModel() const {
            return combined_geom_model_;
        }

    private:
        std::string expandPackagePath(const std::string &path) {
            // Simple package path expansion - replace
            // "package://paper_sequential_planner/" with actual path
            const std::string package_prefix =
                "package://paper_sequential_planner/";
            if (path.find(package_prefix) == 0) {
                std::string base_path = "../rnd_torus/ur5e/";
                return base_path + path.substr(package_prefix.length());
            }
            return path;
        }

        void addCollisionObject(const CollisionObject &collision_obj) {
            // Load collision object model
            pinocchio::Model obj_model;
            pinocchio::urdf::buildModel(collision_obj.urdf_file, obj_model);

            // Load collision object geometry
            pinocchio::GeometryModel obj_geom_model;
            pinocchio::urdf::buildGeom(obj_model,
                                       collision_obj.urdf_file,
                                       pinocchio::COLLISION,
                                       obj_geom_model);

            // Add geometries to combined model with transformed poses
            for (size_t i = 0; i < obj_geom_model.geometryObjects.size(); ++i) {
                pinocchio::GeometryObject obj_geom =
                    obj_geom_model.geometryObjects[i];

                // Apply the pose transformation
                obj_geom.placement = collision_obj.pose * obj_geom.placement;

                // Rename to include object name
                obj_geom.name = collision_obj.name + "_" + obj_geom.name;

                // Add to combined model
                combined_geom_model_.addGeometryObject(obj_geom);
            }
        }

        void setupCollisionPairs() {
            // Add all collision pairs
            combined_geom_model_.addAllCollisionPairs();

            // Remove robot self-collision pairs based on SRDF
            pinocchio::srdf::removeCollisionPairs(
                robot_model_, combined_geom_model_, robot_srdf_file_);

            // Note: Additional filtering for no_collision_links could be
            // implemented here by removing specific collision pairs involving
            // those links
        }
};

int main() {
    using namespace pinocchio;

    // Create collision scene
    UR5eCollisionScene collision_scene;

    // Load configuration from YAML file
    std::string yaml_file = "ur5e_col.yaml";
    if (!collision_scene.loadFromYAML(yaml_file)) {
        std::cerr << "Failed to load YAML configuration file: " << yaml_file
                  << std::endl;
        return -1;
    }

    // Build the collision scene
    if (!collision_scene.buildCollisionScene()) {
        std::cerr << "Failed to build collision scene" << std::endl;
        return -1;
    }

    // Print collision scene information
    collision_scene.printCollisionInfo();

    // Print joint information
    const Model &robot_model = collision_scene.getRobotModel();
    std::cout << "\n=== JOINT INFORMATION ===" << std::endl;
    for (Model::JointIndex j = 0; j < robot_model.njoints; ++j) {
        std::cout << "Joint " << j << " : " << robot_model.names[j]
                  << " has index_qs " << robot_model.idx_qs[j] << " and nqs "
                  << robot_model.nqs[j] << std::endl;
    }

    // Test collision detection for shoulder_pan_joint from 0 to 2π
    std::cout << "\n=== COLLISION TEST FOR SHOULDER PAN JOINT (0 to 2π) ==="
              << std::endl;

    const int num_samples = 36; // Test every 10 degrees (360/10 = 36 samples)
    const double pi = M_PI;

    // Initialize joint configuration vector
    Eigen::VectorXd q = Eigen::VectorXd::Zero(robot_model.nq);

    std::cout << "Testing " << num_samples
              << " configurations from 0 to 2π radians:" << std::endl;
    std::cout << "Pan Joint [rad] | Pan Joint [deg] | Collision?" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    for (int i = 0; i < num_samples; ++i) {
        // Calculate pan joint angle
        double pan_angle = (2.0 * pi * i) / (num_samples - 1);

        // Set shoulder_pan_joint (index 0 in joint configuration)
        q[0] = pan_angle;

        // Check for collisions
        bool hasCollision = collision_scene.checkCollision(q);

        // Convert to degrees for display
        double pan_angle_deg = pan_angle * 180.0 / pi;

        std::printf("%11.3f | %12.1f | %s\n",
                    pan_angle,
                    pan_angle_deg,
                    hasCollision ? "YES" : "no");
    }

    // Also test at exactly 0 and 2π
    std::cout << "\n=== DETAILED COLLISION TEST AT KEY POSITIONS ===" << std::endl;

    std::vector<double> test_angles = {0.0, pi / 2, pi, 3 * pi / 2, 2 * pi};
    std::vector<std::string> angle_names = {"0°", "90°", "180°", "270°", "360°"};

    for (size_t i = 0; i < test_angles.size(); ++i) {
        q[0] = test_angles[i];
        bool hasCollision = collision_scene.checkCollision(q);

        std::cout << "\n--- Testing at " << angle_names[i] << " ("
                  << test_angles[i] << " rad) ---" << std::endl;
        std::cout << "Collision detected: " << (hasCollision ? "YES" : "no")
                  << std::endl;

        if (hasCollision) {
            // Show detailed collision results only if collision is detected
            collision_scene.printDetailedCollisionResults(q);
        }
    }

    return 0;
}