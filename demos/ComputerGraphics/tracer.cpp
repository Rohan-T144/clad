//-----------------------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// This is a modernized and refactored version of a demo showcasing how to use clad
// in a simple path tracer. The original was based on Kevin Beason's 'smallpt'.
//
// This version focuses on:
// - Modern C++ (C++17) features for clarity and safety.
// - Flattened class hierarchy (no polymorphism) for simplicity and clad compatibility.
// - Improved code structure and readability.
// - A focused demonstration of clad for differentiable rendering.
//
// To compile (example using Clang):
// clang++ -std=c++17 -O3 -fopenmp -o ModernPT ModernPT.cpp \
//   -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang /path/to/libclad.so \
//   -I/path/to/clad/include
//
// To run:
// ./ModernPT 16
// (The argument is the number of samples per pixel, e.g., 16)
//-----------------------------------------------------------------------------------------//
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

// --- Core Data Structures and Constants ---
namespace PathTracer {

constexpr double PI = 3.14159265358979323846;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPSILON = 1e-6;

/**
 * @struct Vec
 * @brief Represents a 3D vector for positions, directions, and colors (RGB).
 */
struct Vec {
  double x = 0, y = 0, z = 0;

  Vec() = default;
  constexpr Vec(double x, double y, double z) : x(x), y(y), z(z) {}
  Vec(const Vec& other) : x(other.x), y(other.y), z(other.z) {}
  Vec& operator=(const Vec& other) {
    // if (this != &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    // }
    return *this;
  }

  Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
  Vec operator+=(const Vec& b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
  Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
  Vec operator*=(double b) {
    x *= b;
    y *= b;
    z *= b;
    return *this;
  }
  Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); } // Element-wise multiplication
  Vec& normalize() {
    *this *= (1.0 / std::sqrt(x * x + y * y + z * z));
    return *this;
  }
  double dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; }
  Vec cross(const Vec& b) const { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

/**
 * @struct Ray
 * @brief Represents a ray with an origin and a direction vector.
 */
struct Ray {
  Vec origin, direction;
  Ray() = default;
  Ray(const Vec o, const Vec d) : origin(o), direction(d) {}
  Ray(const Ray& other) : origin(other.origin), direction(other.direction) {}
  // Ray(Ray&& other) noexcept : origin(std::move(other.origin)), direction(std::move(other.direction)) {}
  // Ray& operator=(const Ray& other) {
  //     if (this != &other) {
  //         origin = other.origin;
  //         direction = other.direction;
  //     }
  //     return *this;
  // }
  Ray operator=(const Ray& other) {
    // if (this != &other) {
    origin = other.origin;
    direction = other.direction;
    // }
    return *this;
  }
};

/**
 * @enum MaterialType
 * @brief Defines the type of material for light interaction.
 */
enum MaterialType { DIFFUSE, SPECULAR, REFRACTIVE };

/**
 * @brief Provides a thread-safe, high-quality random number.
 * @return A random double between 0.0 and 1.0.
 */
inline double random_double() {
  thread_local static std::mt19937 generator(std::random_device{}());
  thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(generator);
}

// --- Geometry and Scene Objects ---

/**
 * @brief The implicit function for a sphere, f(p) = 0.
 * The gradient of this function gives the normal vector.
 */
double sphere_implicit_func(const Vec& p, const Vec& center, double radius) {
  return (p.x - center.x) * (p.x - center.x) + (p.y - center.y) * (p.y - center.y) +
         (p.z - center.z) * (p.z - center.z) - radius * radius;
}

class Sphere {
public:
  Vec emission, color, center;
  MaterialType material;
  double radius;
  Sphere() = default;
  Sphere(double r, Vec p, Vec e, Vec c, MaterialType m) : emission(e), color(c), center(p), material(m), radius(r) {}
  void operator+=(const Sphere& other) {
    emission += other.emission;
    color += other.color;
    center += other.center; // Overwrite center, not additive
    radius += other.radius; // Overwrite radius, not additive
                            // material = other.material; // Overwrite material type
  }
  /**
   * @brief Computes ray-sphere intersection using the analytic solution.
   */
  double intersect(const Ray& ray) const {
    Vec op = center - ray.origin;
    double t;
    double b = op.dot(ray.direction);
    double det = b * b - op.dot(op) + radius * radius;

    if (det < 0)
      return INF;

    det = std::sqrt(det);
    t = b - det;
    if (t > EPSILON)
      return t;

    t = b + det;
    if (t > EPSILON)
      return t;

    return INF;
  }
  static double sphere_func_dx(const Vec& p, const Vec& p0, double r) { return 2 * (p.x - p0.x); }

  static double sphere_func_dy(const Vec& p, const Vec& p0, double r) { return 2 * (p.y - p0.y); }

  static double sphere_func_dz(const Vec& p, const Vec& p0, double r) { return 2 * (p.z - p0.z); }
  /**
   * @brief Computes the surface normal using automatic differentiation with clad.
   */
  Vec normal(const Vec& pt) const {
    double Nx = sphere_func_dx(pt, center, radius);
    double Ny = sphere_func_dy(pt, center, radius);
    double Nz = sphere_func_dz(pt, center, radius);
    return Vec(Nx, Ny, Nz).normalize();
    // Vec gradient_result{};
    // Vec dummy_center_grad{}; // clad requires gradients for all specified vars

    // // Differentiate the implicit function with respect to the point `p`.
    // // The gradient of f(p)=0 at a point on the surface is the normal.
    // auto grad_func = clad::gradient(sphere_implicit_func, "p, center");
    // grad_func.execute(point, center, radius, &gradient_result, &dummy_center_grad);

    // return gradient_result.normalize();
  }
};

// --- Scene Management ---

using Scene = std::vector<Sphere>;

struct Intersection {
  double distance = INF;
  size_t object_id = -1;
  bool hit = false;
};

/**
 * @brief Finds the closest object in the scene that intersects with the ray.
 * @param scene The scene to test against.
 * @param ray The ray to trace.
 * @return An Intersection struct with hit details.
 */
Intersection find_intersection(const Scene& scene, const Ray& ray) {
  Intersection closest_hit;
  closest_hit.distance = INF;

  for (size_t i = 0; i < scene.size(); ++i) {
    double d = scene[i].intersect(ray);
    if (d < closest_hit.distance) {
      closest_hit.distance = d;
      closest_hit.object_id = i;
      closest_hit.hit = true;
    }
  }
  return closest_hit;
}

/**
 * @brief Creates the Cornell Box scene.
 * @return A Scene object populated with spheres.
 */
Scene create_scene() {
  Scene scene;
  // The Cornell Box walls, floor, ceiling, and light are large spheres.
  scene.emplace_back(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(0.75, 0.25, 0.25), DIFFUSE);   // Left
  scene.emplace_back(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(0.25, 0.25, 0.75), DIFFUSE); // Right
  scene.emplace_back(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(0.75, 0.75, 0.75), DIFFUSE);         // Back
  scene.emplace_back(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFFUSE);                  // Front
  scene.emplace_back(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(0.75, 0.75, 0.75), DIFFUSE);         // Bottom
  scene.emplace_back(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(0.75, 0.75, 0.75), DIFFUSE); // Top
  scene.emplace_back(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * 0.999, SPECULAR);         // Mirror
  scene.emplace_back(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * 0.999, REFRACTIVE);       // Glass
  scene.emplace_back(600, Vec(50, 681.6 - 0.27, 81.6), Vec(12, 12, 12), Vec(), DIFFUSE);      // Light
  return scene;
}

// --- Core Path Tracing Logic ---

/**
 * @brief Calculates the color (radiance) for a given ray.
 * This function iteratively traces the ray as it bounces through the scene.
 * @param scene The scene to trace within.
 * @param ray The initial ray.
 * @return The computed color for the ray.
 */
Vec radiance(Scene& scene, Ray ray) {
  Vec accumulated_color;
  Vec accumulated_reflectance(1, 1, 1);

  for (int depth = 0;; ++depth) {
    Intersection hit = find_intersection(scene, ray);

    if (!hit.hit)
      return accumulated_color; // Ray missed the scene, return black.

    const Sphere& obj = scene[hit.object_id];
    Vec hit_point = ray.origin + ray.direction * hit.distance;
    Vec normal = obj.normal(hit_point);
    Vec oriented_normal = normal.dot(ray.direction) < 0 ? normal : normal * -1;

    // Add emitted light from the surface
    accumulated_color = accumulated_color + accumulated_reflectance.mult(obj.emission);

    // Russian Roulette for path termination
    // double max_reflectance = std::max({obj.color.x, obj.color.y, obj.color.z});
    // double max_reflectance = std::max(std::max(obj.color.x, obj.color.y), obj.color.z);
    double max_reflectance = obj.color.x;
    if (obj.color.y > obj.color.x)
      max_reflectance = obj.color.y;
    else if (obj.color.z > max_reflectance)
      max_reflectance = obj.color.z;
    if (depth > 5)
      if (random_double() < max_reflectance)
        accumulated_reflectance = accumulated_reflectance.mult(obj.color * (1.0 / max_reflectance));
      else
        return accumulated_color;
    else
      accumulated_reflectance = accumulated_reflectance.mult(obj.color);

    // Handle different material types
    if (obj.material == DIFFUSE) {
      double r1 = 2 * PI * random_double();
      double r2 = random_double();
      double r2s = std::sqrt(r2);

      Vec w = oriented_normal;
      Vec u = (std::abs(w.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0)).cross(w).normalize();
      Vec v = w.cross(u);
      Vec new_dir = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s + w * std::sqrt(1 - r2)).normalize();

      ray = Ray(hit_point, new_dir);
      continue;
    } else if (obj.material == SPECULAR) {
      Vec new_dir = ray.direction - normal * 2 * normal.dot(ray.direction);
      ray = Ray(hit_point, new_dir);
      continue;
    } else if (obj.material == REFRACTIVE) {
      Ray reflection_ray(hit_point, ray.direction - normal * 2 * normal.dot(ray.direction));
      bool into = normal.dot(oriented_normal) > 0; // Is ray entering or leaving the object?
      double nc = 1, nt = 1.5;                     // Refractive indices (air, glass)
      double nnt = into ? nc / nt : nt / nc;
      double ddn = ray.direction.dot(oriented_normal);
      double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

      if (cos2t < 0) { // Total internal reflection
        ray = reflection_ray;
        continue;
      }

      Vec tdir = (ray.direction * nnt - normal * ((into ? 1 : -1) * (ddn * nnt + std::sqrt(cos2t)))).normalize();

      // Fresnel effect to mix reflection and refraction
      double a = nt - nc, b = nt + nc, R0 = (a * a) / (b * b);
      double c = 1 - (into ? -ddn : tdir.dot(normal));
      double pow = std::pow(c, 5);
      double Re = R0 + (1 - R0) * pow;
      double Tr = 1 - Re;
      double P = 0.25 + 0.5 * Re; // Probability of reflection

      if (random_double() < P) {
        accumulated_reflectance = accumulated_reflectance * (Re / P);
        ray = reflection_ray;
      } else {
        accumulated_reflectance = accumulated_reflectance * (Tr / (1.0 - P));
        ray = Ray(hit_point, tdir);
      }
      continue;
    }
  }
}

// --- Rendering and Image Output ---

/**
 * @brief Renders the entire scene.
 * @return A vector of pixel colors representing the final image.
 */
void render(int w, int h, int samples, const Ray& cam, Scene& scene, Vec* image) {
  // std::vector<Vec> image(w * h);
  Vec cx = Vec(w * 0.5135 / h, 0, 0);
  Vec cy = (cx.cross(cam.direction)).normalize() * 0.5135;

  // #pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < h; ++y) {
    // fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samples * 4, 100. * y / (h - 1));
    for (int x = 0; x < w; ++x) {
      // 2x2 subpixel sampling for antialiasing
      for (int sy = 0; sy < 2; ++sy) {
        for (int sx = 0; sx < 2; ++sx) {
          Vec accumulated_radiance;
          for (int s = 0; s < samples; ++s) {
            // Tent filter for smoother sampling
            double r1 = 2 * random_double(), dx = r1 < 1 ? std::sqrt(r1) - 1 : 1 - std::sqrt(2 - r1);
            double r2 = 2 * random_double(), dy = r2 < 1 ? std::sqrt(r2) - 1 : 1 - std::sqrt(2 - r2);

            Vec d =
              cx * (((sx + 0.5 + dx) / 2 + x) / w - 0.5) + cy * (((sy + 0.5 + dy) / 2 + y) / h - 0.5) + cam.direction;

            // Start rays inside the box to avoid hitting the camera sphere
            Ray sample_ray(cam.origin + d * 140, d.normalize());
            accumulated_radiance = accumulated_radiance + radiance(scene, sample_ray) * (1.0 / samples);
          }
          int i = (h - 1 - y) * w + x;
          image[i] = image[i] + accumulated_radiance * 0.25;
        }
      }
    }
  }
  // fprintf(stderr, "\n");
  // return image;
}

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int to_int(double x) { return static_cast<int>(std::pow(clamp(x), 1 / 2.2) * 255 + 0.5); }

/**
 * @brief Saves the rendered image to a PPM file.
 */
void save_ppm(const std::string& filename, int w, int h, const std::vector<Vec>& image) {
  FILE* f = fopen(filename.c_str(), "wb");
  if (!f) {
    perror("fopen");
    return;
  }
  fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
  for (const auto& pixel : image) {
    fputc(to_int(pixel.x), f);
    fputc(to_int(pixel.y), f);
    fputc(to_int(pixel.z), f);
  }
  fclose(f);
}

// --- Differentiable Rendering with Clad ---

/**
 * @brief A loss function for differentiable rendering.
 * It computes the sum of squared differences between a rendered image and a target.
 * Clad can differentiate this function to find how scene parameters affect the final image.
 *
 * @param w, h, samples Render settings.
 * @param cam The camera, whose parameters we might want to optimize.
 * @param scene The scene definition.
 * @param target_image The ground truth image to compare against.
 * @return The computed loss value.
 */
double loss_function(int w, int h, int samples, const Ray& cam, Scene& scene, const Vec* target_image) {
  Vec* rendered_image = new Vec[w * h];
  // Render the image using the provided camera and scene.
  render(w, h, samples, cam, scene, rendered_image);

  // Calculate the loss as the sum of squared differences between the rendered image and target image.
  double loss = 0.0;
  for (size_t i = 0; i < w * h; ++i) {
    Vec diff = rendered_image[i] - target_image[i];
    loss += diff.dot(diff);
  }
  delete[] rendered_image; // Clean up allocated memory
  return loss;
}

} // namespace PathTracer

namespace clad::custom_derivatives {
namespace PathTracer {}
namespace class_functions {
// void operator_minus_pullback(
//   const ::PathTracer::Vec* _this, const ::PathTracer::Vec& b, ::PathTracer::Vec _d_y, ::PathTracer::Vec* _d_this, ::PathTracer::Vec* _d_other
// ) {
//   *_d_this += _d_y; // Gradient flows to the left operand
//   *_d_other = *_d_other - _d_y; // Gradient flows to the right operand
// }
void operator_equal_pullback(
  const ::PathTracer::Vec* _this, const ::PathTracer::Vec& other, const ::PathTracer::Vec _d_y,
  const ::PathTracer::Vec* _d_this, ::PathTracer::Vec* _d_other
) {
  // For assignment, the gradient flows to both tensors
  *_d_other += *_d_this;
}
void operator_equal_pullback(
  const ::PathTracer::Ray* _this, const ::PathTracer::Ray& other, const ::PathTracer::Ray _d_y,
  const ::PathTracer::Ray* _d_this, ::PathTracer::Ray* _d_other
) {
  // For assignment, the gradient flows to both tensors
  _d_other->direction += _d_this->direction;
  _d_other->origin += _d_this->origin;
}
} // namespace class_functions
} // namespace clad::custom_derivatives

int main(int argc, char* argv[]) {
  using namespace PathTracer;

  int w = 256, h = 192;
  int samples = (argc == 2) ? std::atoi(argv[1]) : 4;

  // --- Setup Scene and Camera ---
  auto scene = create_scene();
  Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).normalize());

  // --- Render the Image ---
  std::cout << "Rendering image..." << std::endl;
  std::vector<Vec> image(w * h);
  render(w, h, samples, cam, scene, image.data());
  save_ppm("image.ppm", w, h, image);
  std::cout << "Image saved to image.ppm" << std::endl;

  // --- Demonstrate Clad ---
  // The demonstration part is left here but commented out to match user edits.
  // In a real application, you would use this gradient in an optimization
  // loop (e.g., gradient descent) to update camera or scene parameters.
  // std::cout << "\nDemonstrating clad for automatic differentiation..." << std::endl;

  auto grad_func = clad::gradient(loss_function);
  grad_func.dump();

  // Ray cam_grad_result{Vec(), Vec()};
  // double loss_val = grad_func.execute(w, h, samples, cam, scene, target_image, &cam_grad_result);

  // std::cout << "Clad analysis complete." << std::endl;
  // std::cout << "Loss value: " << loss_val << std::endl;
  // std::cout << "Gradient of camera origin: ("
  //           << cam_grad_result.origin.x << ", "
  //           << cam_grad_result.origin.y << ", "
  //           << cam_grad_result.origin.z << ")" << std::endl;

  return 0;
}
