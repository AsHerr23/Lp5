#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <omp.h>

using namespace std;

struct Point {
    double x, y;
    int cluster;
};

double distance(Point a, Point b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

int main() {
    int num_points = 100000;
    int k = 3;
    int max_iters = 100;

    srand(time(0));

    // Generate random points
    vector<Point> points(num_points);
    for (auto &p : points) {
        p.x = rand() % 1000;
        p.y = rand() % 1000;
        p.cluster = -1;
    }

    // Initialize random centroids
    vector<Point> centroids(k);
    for (int i = 0; i < k; ++i) {
        centroids[i] = points[rand() % num_points];
    }

    bool changed = true;
    int iters = 0;

    while (changed && iters++ < max_iters) {
        changed = false;

        // Assign clusters (Parallelized)
        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            double min_dist = numeric_limits<double>::max();
            int cluster_id = -1;
            for (int j = 0; j < k; ++j) {
                double dist = distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster_id = j;
                }
            }
            if (points[i].cluster != cluster_id) {
                points[i].cluster = cluster_id;
                changed = true;
            }
        }

        // Recompute centroids
        vector<double> sum_x(k, 0.0), sum_y(k, 0.0);
        vector<int> count(k, 0);

        #pragma omp parallel for reduction(+:sum_x[:k], sum_y[:k], count[:k])
        for (int i = 0; i < num_points; ++i) {
            int c = points[i].cluster;
            sum_x[c] += points[i].x;
            sum_y[c] += points[i].y;
            count[c]++;
        }

        for (int j = 0; j < k; ++j) {
            if (count[j] != 0) {
                centroids[j].x = sum_x[j] / count[j];
                centroids[j].y = sum_y[j] / count[j];
            }
        }
    }

    cout << "K-Means completed in " << iters << " iterations.\n";
    for (int i = 0; i < k; ++i) {
        cout << "Centroid " << i << ": (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }

    return 0;
}


