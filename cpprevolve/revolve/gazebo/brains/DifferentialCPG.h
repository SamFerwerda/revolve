/*
 * Copyright (C) 2015-2018 Vrije Universiteit Amsterdam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Description: TODO: <Add brief description about file purpose>
 * Author: Milan Jelisavcic
 * Date: December 29, 2018
 *
 */

#ifndef REVOLVE_DIFFERENTIALCPG_H_
#define REVOLVE_DIFFERENTIALCPG_H_

// Standard libraries
#include <map>
#include <tuple>

// External libraries
#include <eigen3/Eigen/Core>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>

// Project headers
#include "Evaluator.h"
#include "Brain.h"

/// These numbers are quite arbitrary. It used to be in:13 out:8 for the
/// Arduino, but I upped them both to 20 to accommodate other scenarios.
/// Should really be enforced in the Python code, this implementation should
/// not be the limit.
#define MAX_INPUT_NEURONS 20
#define MAX_OUTPUT_NEURONS 20

/// Arbitrary value
#define MAX_HIDDEN_NEURONS 30

/// Convenience
#define MAX_NON_INPUT_NEURONS (MAX_HIDDEN_NEURONS + MAX_OUTPUT_NEURONS)

/// (bias, tau, gain) or (phase offset, period, gain)
#define MAX_NEURON_PARAMS 3




typedef std::vector< double > state_type;

namespace revolve
{
  namespace gazebo {
    class DifferentialCPG
        : public Brain
    {
      /// \brief Constructor
      /// \param[in] _modelName Name of the robot
      /// \param[in] _node The brain node
      /// \param[in] _motors Reference to a motor list, it be reordered
      /// \param[in] _sensors Reference to a sensor list, it might be reordered
      public:
      DifferentialCPG(
          const ::gazebo::physics::ModelPtr &_model,
          const sdf::ElementPtr _settings,
          const std::vector< MotorPtr > &_motors,
          const std::vector< SensorPtr > &_sensors);

      public:
      void set_ODE_matrix();

      /// \brief Destructor
      public:
      virtual ~DifferentialCPG();

      /// \brief The default update method for the controller
      /// \param[in] _motors Motor list
      /// \param[in] _sensors Sensor list
      /// \param[in] _time Current world time
      /// \param[in] _step Current time step
      public:
      virtual void Update(
          const std::vector< MotorPtr > &_motors,
          const std::vector< SensorPtr > &_sensors,
          const double _time,
          const double _step);

      protected:
      void Step(
          const double _time,
          double *_output);

      /// \brief Register of motor IDs and their x,y-coordinates
      protected:
      std::map< std::string, std::tuple< int, int > > positions;

      /// \brief Register of individual neurons in x,y,z-coordinates
      /// \details x,y-coordinates define position of a robot's module and
      // z-coordinate define A or B neuron (z=1 or -1 respectively). Stored
      // values are a bias, gain, state of each neuron.
      protected:
      std::map< std::tuple< int, int, int >, std::tuple< double, double, double > >
          neurons;

      /// \brief Register of connections between neighnouring neurons
      /// \details Coordinate set of two neurons (x1, y1, z1) and (x2, y2, z2)
      // define a connection. The second tuple contains 1: the connection value and
      // 2: the weight index corresponding to this connection.
      protected:
      std::map< std::tuple< int, int, int, int, int, int >, std::tuple<int, int > >
          connections;

      /// \brief Runge-Kutta 45 stepper
      protected: boost::numeric::odeint::runge_kutta4< state_type > stepper;

      /// \brief Used to determine the next state array
      private: double *next_state;

      /// \brief Used for ODE-int
      protected: std::vector<std::vector<double>> ode_matrix;
      protected: state_type x;

      /// \brief One input state for each input neuron
      private: double *input;

      /// \brief Used to determine the output to the motors array
      private: double *output;

      /// \brief Location where to save output
      private: std::string directory_name;

      /// \brief Name of the robot
      private: ::gazebo::physics::ModelPtr robot;

      /// \brief Init BO loop
      public: void BO_init();

      /// \brief Main BO loop
      public: void BO_step();

      /// \brief evaluation rate
      private: double evaluation_rate;

      /// \brief Get fitness
      private: void save_fitness();

      /// \brief Pointer to the fitness evaluator
      protected: EvaluatorPtr evaluator;

      /// \brief Holder for BO parameters
      public: struct Params;

      /// \brief Best fitness seen so far
      private: double best_fitness;

      /// \brief Sample corresponding to best fitness
      private: Eigen::VectorXd best_sample;

      /// \brief Starting time
      private: double start_time;

      /// \brief BO attributes
      private: size_t current_iteration = 0;

      /// \brief Max number of iterations learning is allowed
      private: size_t max_learning_iterations;

      /// \brief Number of initial samples
      private: size_t n_init_samples;

      /// \brief Cool down period
      private: size_t no_learning_iterations;

      /// \brief Limbo optimizes in [0,1]
      private: double range_lb;

      /// \brief Limbo optimizes in [0,1]
      private: double range_ub;

      /// \brief How to take initial random samples
      private: std::string init_method;

      /// \brief All fitnesses seen so far. Called observations in limbo context
      private: std::vector< Eigen::VectorXd > observations;

      /// \brief All samples seen so far.
      private: std::vector< Eigen::VectorXd > samples;

      /// \brief The number of weights to optimize
      private: int n_weights;

      /// \brief Dummy evaluation funtion to reduce changes to be made on the limbo package
      public: struct evaluationFunction;

      /// \brief Boolean to enable/disable constructing plots
      private: bool run_analytics;

      /// \brief Automatically generate plots
      public: void get_analytics();

      /// \brief absolute bound on motor signal value
      public: double abs_output_bound;

      /// \brief Holds the number of motors in the robot
      private: size_t n_motors;

      /// \brief Helper for numerical integrator
      private: double previous_time = 0;

      /// \brief Initial neuron state
      private: double init_state = M_SQRT2/2.f
      ;
    };
  }
}

#endif //REVOLVE_DIFFERENTIALCPG_H_
