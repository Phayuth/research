#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

struct City {
        double x, y, z;
        int id;

        City(double x = 0, double y = 0, double z = 0, int id = 0)
            : x(x), y(y), z(z), id(id) {
        }
};

class TSPSolver {
    private:
        std::vector<City> cities;
        std::vector<std::vector<double>> distanceMatrix;

        // Calculate Euclidean distance between two cities
        double calculateDistance(const City &a, const City &b) {
            double dx = a.x - b.x;
            double dy = a.y - b.y;
            double dz = a.z - b.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        // Build distance matrix for all city pairs
        void buildDistanceMatrix() {
            int n = cities.size();
            distanceMatrix.resize(n, std::vector<double>(n, 0.0));

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        distanceMatrix[i][j] =
                            calculateDistance(cities[i], cities[j]);
                    } else {
                        distanceMatrix[i][j] = 0.0;
                    }
                }
            }
        }

    public:
        // Add a city to the problem
        void addCity(double x, double y, double z, int id = -1) {
            if (id == -1) {
                id = cities.size();
            }
            cities.push_back(City(x, y, z, id));
        }

        // Load cities from a file (format: x y z per line)
        bool loadCitiesFromFile(const std::string &filename) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open file " << filename << std::endl;
                return false;
            }

            double x, y, z;
            int id = 0;
            while (file >> x >> y >> z) {
                addCity(x, y, z, id++);
            }

            file.close();
            std::cout << "Loaded " << cities.size() << " cities from " << filename
                      << std::endl;
            return true;
        }

        // Nearest Neighbor Approximation Algorithm
        std::pair<std::vector<int>, double>
        solveNearestNeighbor(int startCity = 0) {
            if (cities.empty()) {
                std::cerr << "Error: No cities loaded!" << std::endl;
                return {{}, 0.0};
            }

            buildDistanceMatrix();

            int n = cities.size();
            std::vector<bool> visited(n, false);
            std::vector<int> tour;
            double totalDistance = 0.0;

            // Start from the specified city
            int currentCity = startCity;
            tour.push_back(currentCity);
            visited[currentCity] = true;

            // Visit all other cities
            for (int step = 1; step < n; step++) {
                int nearestCity = -1;
                double nearestDistance = std::numeric_limits<double>::max();

                // Find the nearest unvisited city
                for (int i = 0; i < n; i++) {
                    if (!visited[i] &&
                        distanceMatrix[currentCity][i] < nearestDistance) {
                        nearestDistance = distanceMatrix[currentCity][i];
                        nearestCity = i;
                    }
                }

                // Move to the nearest city
                if (nearestCity != -1) {
                    tour.push_back(nearestCity);
                    visited[nearestCity] = true;
                    totalDistance += nearestDistance;
                    currentCity = nearestCity;
                }
            }

            // Return to the starting city
            totalDistance += distanceMatrix[currentCity][startCity];
            tour.push_back(startCity);

            return {tour, totalDistance};
        }

        // Try all possible starting cities and return the best tour
        std::pair<std::vector<int>, double> solveBestNearestNeighbor() {
            if (cities.empty()) {
                std::cerr << "Error: No cities loaded!" << std::endl;
                return {{}, 0.0};
            }

            std::vector<int> bestTour;
            double bestDistance = std::numeric_limits<double>::max();

            // Try starting from each city
            for (int start = 0; start < cities.size(); start++) {
                auto [tour, distance] = solveNearestNeighbor(start);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestTour = tour;
                }
            }

            return {bestTour, bestDistance};
        }

        // Print the tour
        void printTour(const std::vector<int> &tour, double totalDistance) {
            std::cout << "\n=== TSP Solution (Nearest Neighbor) ===" << std::endl;
            std::cout << "Tour length: " << std::fixed << std::setprecision(3)
                      << totalDistance << std::endl;
            std::cout << "Tour sequence:" << std::endl;

            for (size_t i = 0; i < tour.size(); i++) {
                int cityIdx = tour[i];
                std::cout << "City " << cities[cityIdx].id << " (" << std::fixed
                          << std::setprecision(2) << cities[cityIdx].x << ", "
                          << cities[cityIdx].y << ", " << cities[cityIdx].z << ")";
                if (i < tour.size() - 1) {
                    std::cout << " -> ";
                }
                std::cout << std::endl;
            }
        }

        // Save tour to file
        void saveTourToFile(const std::vector<int> &tour, double totalDistance,
                            const std::string &filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot create file " << filename << std::endl;
                return;
            }

            file << "# TSP Solution using Nearest Neighbor Approximation"
                 << std::endl;
            file << "# Total distance: " << std::fixed << std::setprecision(6)
                 << totalDistance << std::endl;
            file << "# Format: city_id x y z" << std::endl;

            for (int cityIdx : tour) {
                file << cities[cityIdx].id << " " << std::fixed
                     << std::setprecision(6) << cities[cityIdx].x << " "
                     << cities[cityIdx].y << " " << cities[cityIdx].z << std::endl;
            }

            file.close();
            std::cout << "Tour saved to " << filename << std::endl;
        }

        // Get number of cities
        size_t getNumCities() const {
            return cities.size();
        }

        // Clear all cities
        void clear() {
            cities.clear();
            distanceMatrix.clear();
        }

        // Calculate tour distance given a tour sequence
        double calculateTourDistance(const std::vector<int> &tour) {
            if (tour.size() < 2)
                return 0.0;

            double totalDistance = 0.0;
            for (size_t i = 0; i < tour.size() - 1; i++) {
                totalDistance += distanceMatrix[tour[i]][tour[i + 1]];
            }
            return totalDistance;
        }

        // 2-opt improvement algorithm
        std::pair<std::vector<int>, double> improve2Opt(std::vector<int> tour) {
            if (cities.empty() || tour.size() < 4) {
                return {tour, calculateTourDistance(tour)};
            }

            // Remove the duplicate last city for processing
            if (tour.front() == tour.back()) {
                tour.pop_back();
            }

            int n = tour.size();
            bool improved = true;
            double bestDistance = calculateTourDistance(tour) +
                                  distanceMatrix[tour.back()][tour.front()];

            while (improved) {
                improved = false;

                for (int i = 0; i < n - 1; i++) {
                    for (int j = i + 2; j < n; j++) {
                        if (j == n - 1 && i == 0)
                            continue; // Skip if it would disconnect the tour

                        // Calculate current distance
                        int city1 = tour[i];
                        int city2 = tour[i + 1];
                        int city3 = tour[j];
                        int city4 = tour[(j + 1) % n];

                        double currentDist = distanceMatrix[city1][city2] +
                                             distanceMatrix[city3][city4];
                        double newDist = distanceMatrix[city1][city3] +
                                         distanceMatrix[city2][city4];

                        if (newDist < currentDist) {
                            // Reverse the segment between i+1 and j
                            std::reverse(tour.begin() + i + 1,
                                         tour.begin() + j + 1);
                            improved = true;
                            bestDistance = bestDistance - currentDist + newDist;
                        }
                    }
                }
            }

            // Add the starting city back at the end to complete the cycle
            tour.push_back(tour[0]);
            return {tour, bestDistance};
        }

        // 3-opt improvement (basic version)
        std::pair<std::vector<int>, double> improve3Opt(std::vector<int> tour) {
            if (cities.empty() || tour.size() < 6) {
                return improve2Opt(tour);
            }

            // Remove the duplicate last city for processing
            if (tour.front() == tour.back()) {
                tour.pop_back();
            }

            int n = tour.size();
            bool improved = true;
            double bestDistance = calculateTourDistance(tour) +
                                  distanceMatrix[tour.back()][tour.front()];

            while (improved) {
                improved = false;

                for (int i = 0; i < n - 2; i++) {
                    for (int j = i + 2; j < n - 1; j++) {
                        for (int k = j + 2; k < n; k++) {
                            if (k == n - 1 && i == 0)
                                continue;

                            // Current edges
                            double currentDist =
                                distanceMatrix[tour[i]][tour[i + 1]] +
                                distanceMatrix[tour[j]][tour[j + 1]] +
                                distanceMatrix[tour[k]][tour[(k + 1) % n]];

                            // Try different reconnections (simplified 3-opt)
                            std::vector<std::vector<int>> candidates;

                            // Reconnection 1: reverse segment (i+1, j)
                            std::vector<int> tour1 = tour;
                            std::reverse(tour1.begin() + i + 1,
                                         tour1.begin() + j + 1);

                            // Reconnection 2: reverse segment (j+1, k)
                            std::vector<int> tour2 = tour;
                            std::reverse(tour2.begin() + j + 1,
                                         tour2.begin() + k + 1);

                            candidates.push_back(tour1);
                            candidates.push_back(tour2);

                            for (const auto &candidate : candidates) {
                                double newDist =
                                    distanceMatrix[candidate[i]]
                                                  [candidate[i + 1]] +
                                    distanceMatrix[candidate[j]]
                                                  [candidate[j + 1]] +
                                    distanceMatrix[candidate[k]]
                                                  [candidate[(k + 1) % n]];

                                if (newDist < currentDist) {
                                    tour = candidate;
                                    improved = true;
                                    bestDistance =
                                        bestDistance - currentDist + newDist;
                                    break;
                                }
                            }

                            if (improved)
                                break;
                        }
                        if (improved)
                            break;
                    }
                    if (improved)
                        break;
                }
            }

            // Add the starting city back at the end
            tour.push_back(tour[0]);
            return {tour, bestDistance};
        }

        // Or-opt improvement (relocate segments)
        std::pair<std::vector<int>, double> improveOrOpt(std::vector<int> tour) {
            if (cities.empty() || tour.size() < 5) {
                return {tour, calculateTourDistance(tour)};
            }

            // Remove the duplicate last city for processing
            if (tour.front() == tour.back()) {
                tour.pop_back();
            }

            int n = tour.size();
            bool improved = true;
            double bestDistance = calculateTourDistance(tour) +
                                  distanceMatrix[tour.back()][tour.front()];

            while (improved) {
                improved = false;

                // Try relocating segments of length 1, 2, and 3
                for (int segLen = 1; segLen <= std::min(3, n - 3); segLen++) {
                    for (int i = 0; i < n - segLen; i++) {
                        for (int j = 0; j < n; j++) {
                            if (j >= i && j <= i + segLen)
                                continue; // Skip overlapping positions

                            // Calculate current cost
                            double currentCost = 0.0;
                            if (i > 0)
                                currentCost +=
                                    distanceMatrix[tour[i - 1]][tour[i]];
                            if (i + segLen < n)
                                currentCost += distanceMatrix[tour[i + segLen - 1]]
                                                             [tour[i + segLen]];
                            if (i > 0 && i + segLen < n)
                                currentCost -=
                                    distanceMatrix[tour[i - 1]][tour[i + segLen]];

                            // Calculate new cost
                            double newCost = 0.0;
                            if (j > 0)
                                newCost += distanceMatrix[tour[j - 1]][tour[i]];
                            if (j < n)
                                newCost +=
                                    distanceMatrix[tour[i + segLen - 1]][tour[j]];
                            if (j > 0 && j < n)
                                newCost -= distanceMatrix[tour[j - 1]][tour[j]];

                            if (newCost < currentCost) {
                                // Perform the relocation
                                std::vector<int> segment(
                                    tour.begin() + i, tour.begin() + i + segLen);
                                tour.erase(tour.begin() + i,
                                           tour.begin() + i + segLen);

                                int insertPos = j;
                                if (j > i)
                                    insertPos -= segLen;

                                tour.insert(tour.begin() + insertPos,
                                            segment.begin(),
                                            segment.end());
                                improved = true;
                                bestDistance =
                                    bestDistance - currentCost + newCost;
                                break;
                            }
                        }
                        if (improved)
                            break;
                    }
                    if (improved)
                        break;
                }
            }

            // Add the starting city back at the end
            tour.push_back(tour[0]);
            return {tour, bestDistance};
        }

        // Solve with nearest neighbor + 2-opt improvement
        std::pair<std::vector<int>, double>
        solveWithImprovements(int startCity = 0, bool use3opt = false,
                              bool useOrOpt = false) {
            // Start with nearest neighbor
            auto [tour, distance] = solveNearestNeighbor(startCity);

            std::cout << "Initial nearest neighbor distance: " << std::fixed
                      << std::setprecision(3) << distance << std::endl;

            // Apply 2-opt improvement
            auto [improved2opt, distance2opt] = improve2Opt(tour);
            std::cout << "After 2-opt improvement: " << std::fixed
                      << std::setprecision(3) << distance2opt
                      << " (improvement: " << std::fixed << std::setprecision(3)
                      << (distance - distance2opt) << ")" << std::endl;

            tour = improved2opt;
            distance = distance2opt;

            // Apply 3-opt if requested
            if (use3opt) {
                auto [improved3opt, distance3opt] = improve3Opt(tour);
                std::cout << "After 3-opt improvement: " << std::fixed
                          << std::setprecision(3) << distance3opt
                          << " (improvement: " << std::fixed
                          << std::setprecision(3) << (distance - distance3opt)
                          << ")" << std::endl;
                tour = improved3opt;
                distance = distance3opt;
            }

            // Apply Or-opt if requested
            if (useOrOpt) {
                auto [improvedOrOpt, distanceOrOpt] = improveOrOpt(tour);
                std::cout << "After Or-opt improvement: " << std::fixed
                          << std::setprecision(3) << distanceOrOpt
                          << " (improvement: " << std::fixed
                          << std::setprecision(3) << (distance - distanceOrOpt)
                          << ")" << std::endl;
                tour = improvedOrOpt;
                distance = distanceOrOpt;
            }

            return {tour, distance};
        }

        // Solve with all improvements trying different starting points
        std::pair<std::vector<int>, double>
        solveBestWithImprovements(bool use3opt = false, bool useOrOpt = false) {
            if (cities.empty()) {
                std::cerr << "Error: No cities loaded!" << std::endl;
                return {{}, 0.0};
            }

            std::vector<int> bestTour;
            double bestDistance = std::numeric_limits<double>::max();
            int bestStart = 0;

            std::cout << "\nTrying different starting cities with improvements..."
                      << std::endl;

            // Try starting from each city
            for (size_t start = 0; start < cities.size(); start++) {
                std::cout << "\n--- Starting from city " << start << " ---"
                          << std::endl;
                auto [tour, distance] =
                    solveWithImprovements(start, use3opt, useOrOpt);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestTour = tour;
                    bestStart = start;
                }
            }

            std::cout << "\nBest solution found starting from city " << bestStart
                      << std::endl;
            return {bestTour, bestDistance};
        }
};

// Example usage and test function
void runExample() {
    TSPSolver solver;

    // Add some example cities in 3D space
    solver.addCity(0, 0, 0, 0); // City 0
    solver.addCity(1, 2, 1, 1); // City 1
    solver.addCity(3, 1, 2, 2); // City 2
    solver.addCity(2, 3, 0, 3); // City 3
    solver.addCity(4, 0, 1, 4); // City 4
    solver.addCity(1, 4, 3, 5); // City 5

    std::cout << "Solving TSP for " << solver.getNumCities() << " cities..."
              << std::endl;

    // Solve using nearest neighbor starting from city 0
    std::cout << "\n=== Basic Nearest Neighbor ===" << std::endl;
    auto [tour, distance] = solver.solveNearestNeighbor(0);
    solver.printTour(tour, distance);

    std::cout << "\n" << std::string(60, '=') << std::endl;

    // Solve with 2-opt improvement
    std::cout << "\n=== Nearest Neighbor + 2-opt ===" << std::endl;
    auto [tour2opt, distance2opt] = solver.solveWithImprovements(0, false, false);
    solver.printTour(tour2opt, distance2opt);

    std::cout << "\n" << std::string(60, '=') << std::endl;

    // Solve with all improvements
    std::cout << "\n=== Nearest Neighbor + 2-opt + 3-opt + Or-opt ==="
              << std::endl;
    auto [tourAll, distanceAll] = solver.solveWithImprovements(0, true, true);
    solver.printTour(tourAll, distanceAll);

    std::cout << "\n" << std::string(60, '=') << std::endl;

    // Find best solution with all improvements
    std::cout << "\n=== Best Solution with All Improvements ===" << std::endl;
    auto [bestTour, bestDistance] = solver.solveBestWithImprovements(true, true);
    solver.printTour(bestTour, bestDistance);

    // Save the best tour to file
    solver.saveTourToFile(bestTour, bestDistance, "tsp_solution_improved.txt");
}

int main(int argc, char *argv[]) {
    TSPSolver solver;

    if (argc > 1) {
        // Load cities from file if filename is provided
        std::string filename = argv[1];
        bool useImprovements = (argc > 2 && std::string(argv[2]) == "--improve");
        bool useAllImprovements =
            (argc > 2 && std::string(argv[2]) == "--improve-all");

        if (solver.loadCitiesFromFile(filename)) {
            if (useAllImprovements) {
                std::cout << "\nSolving with all improvements (2-opt + 3-opt + "
                             "Or-opt)..."
                          << std::endl;
                auto [bestTour, bestDistance] =
                    solver.solveBestWithImprovements(true, true);
                solver.printTour(bestTour, bestDistance);

                std::string outputFile = "tsp_solution_improved_" + filename;
                solver.saveTourToFile(bestTour, bestDistance, outputFile);
            } else if (useImprovements) {
                std::cout << "\nSolving with 2-opt improvements..." << std::endl;
                auto [bestTour, bestDistance] =
                    solver.solveBestWithImprovements(false, false);
                solver.printTour(bestTour, bestDistance);

                std::string outputFile = "tsp_solution_2opt_" + filename;
                solver.saveTourToFile(bestTour, bestDistance, outputFile);
            } else {
                // Basic nearest neighbor only
                auto [bestTour, bestDistance] = solver.solveBestNearestNeighbor();
                solver.printTour(bestTour, bestDistance);

                std::string outputFile = "tsp_solution_" + filename;
                solver.saveTourToFile(bestTour, bestDistance, outputFile);
            }
        }
    } else {
        // Run example with predefined cities
        std::cout << "Running example with predefined cities..." << std::endl;
        std::cout << "Usage: " << argv[0]
                  << " <cities_file.txt> [--improve|--improve-all]" << std::endl;
        std::cout << "  --improve      : Use 2-opt improvement" << std::endl;
        std::cout
            << "  --improve-all  : Use all improvements (2-opt + 3-opt + Or-opt)"
            << std::endl;
        std::cout << "File format: x y z (one city per line)" << std::endl
                  << std::endl;

        runExample();
    }

    return 0;
}