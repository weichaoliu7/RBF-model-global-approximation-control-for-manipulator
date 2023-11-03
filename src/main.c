#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include "sine.h"
#include "cosine.h"
#include "inv_matrix.h"

// reference: [1]Liu JinKun. Robot Control System Design and MATLAB Simulation[M]. Tsinghua University Press, 2008.
// [2]Lewis F L, Liu K, Yesildirek A. Neural net robot controller with guaranteed tracking performance[J]. IEEE transactions on neural networks, 1995, 6(3): 703-715.

// global variables declaration
#define H 7 // input layer neurons number
#define OUT 2        // output layer neurons number
#define CONTROLLAW 3 // control law selection
#define NumberOfVariable 5
#define ARRAY_SIZE 40000 // sampling times

static double Ts = 0.001; // sampling period
static double t0 = 0.0;   // start time
static double t1 = 40.0;  // end time

static double center[NumberOfVariable][H] = {{-1.5, -1, -0.5, 0, 0.5, 1, 1.5},
                                             {-1.5, -1, -0.5, 0, 0.5, 1, 1.5},
                                             {-1.5, -1, -0.5, 0, 0.5, 1, 1.5},
                                             {-1.5, -1, -0.5, 0, 0.5, 1, 1.5},
                                             {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}}; // RBF function center
static double width = 10;                                                       // RBF function width
static double p[] = {2.9, 0.76, 0.87, 3.04, 0.87};
static double g = 9.8; // gravitational acceleration

double phi1[H], phi2[H];
double weight1[H], weight2[H];
double derivative_weight1[H], derivative_weight2[H];

struct _archive{
    double q1_archive[ARRAY_SIZE];
    double dq1_archive[ARRAY_SIZE];
    double q2_archive[ARRAY_SIZE];
    double dq2_archive[ARRAY_SIZE];
    double error1_archive[ARRAY_SIZE];
    double error2_archive[ARRAY_SIZE];
    double error1_velocity_archive[ARRAY_SIZE];
    double error2_velocity_archive[ARRAY_SIZE];
    double tol1_archive[ARRAY_SIZE];
    double tol2_archive[ARRAY_SIZE];
    double f_estimate_norm_archive[ARRAY_SIZE];
    double f_norm_archive[ARRAY_SIZE];
} archive;

Data q1_desired, dq1_desired, ddq1_desired;
Data q2_desired, dq2_desired, ddq2_desired;

struct Amp{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct M0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

struct B0{
    double q1_desired;
    double dq1_desired;
    double ddq1_desired;
    double q2_desired;
    double dq2_desired;
    double ddq2_desired;
};

void SystemInput(Data *q1_desired, Data *dq1_desired, Data *ddq1_desired, Data *q2_desired, Data *dq2_desired, Data *ddq2_desired, double Ts, double t0, double t1){

    struct Amp amp; // amplitude
    amp.q1_desired = 1;
    amp.dq1_desired = 1;
    amp.ddq1_desired = -1;
    amp.q2_desired = 1;
    amp.dq2_desired = -1;
    amp.ddq2_desired = -1;

    struct M0 m0; // angular frequency
    m0.q1_desired = 1;
    m0.dq1_desired = 1;
    m0.ddq1_desired = 1;
    m0.q2_desired = 1;
    m0.dq2_desired = 1;
    m0.ddq2_desired = 1;

    struct B0 b0; // vertical shift
    b0.q1_desired = 0;
    b0.dq1_desired = 0;
    b0.ddq1_desired = 0;
    b0.q2_desired = 0;
    b0.dq2_desired = 0;
    b0.ddq2_desired = 0;

    sine(q1_desired, Ts, t0, t1, amp.q1_desired, m0.q1_desired, b0.q1_desired);           // desired angular displacement of link 1
    cosine(dq1_desired, Ts, t0, t1, amp.dq1_desired, m0.dq1_desired, b0.dq1_desired);     // desired angular velocity of link 1
    sine(ddq1_desired, Ts, t0, t1, amp.ddq1_desired, m0.ddq1_desired, b0.ddq1_desired);   // desired angular acceleration of link 1
    cosine(q2_desired, Ts, t0, t1, amp.q2_desired, m0.q2_desired, b0.q2_desired);         // desired angular displacement of link 2
    sine(dq2_desired, Ts, t0, t1, amp.dq2_desired, m0.dq2_desired, b0.dq2_desired);       // desired angular velocity of link 2
    cosine(ddq2_desired, Ts, t0, t1, amp.ddq2_desired, m0.ddq2_desired, b0.ddq2_desired); // desired angular acceleration of link 2
}

struct _system_state{
    double q1;   // actual angular displacement of link 1
    double dq1;  // actual angular velocity of link 1
    double ddq1; // actual angular acceleration of link 1
    double q2;   // actual angular displacement of link 2
    double dq2;  // actual angular velocity of link 2
    double ddq2; // actual angular acceleration of link 2
} system_state;

struct _torque{
    double tol1;   // control torque of link 1
    double tol2;   // control torque of link 2
    double tol1_d; // bounded unknown disturbance of link 1
    double tol2_d; // bounded unknown disturbance of link 2
} torque;

struct _dynamics{
    double M[OUT][OUT];  // inertia matrix
    double Vm[OUT][OUT]; // corioliskentripetal matrix
    double G[OUT];       // gravity matrix
    double F[OUT];       // friction
} dynamics;

struct _controller{
    double controller_u1;
    double controller_u2;
    double controller_u3;
    double controller_u4;
    double controller_u5;
    double controller_u6;
    double controller_u7;
    double controller_u8;
    double controller_u9;
    double controller_u10;
    double controller_out1;
    double controller_out2;
    double controller_out3;
    double error1;          // angular displacement error of link 1
    double error1_velocity; // angular velocity error of link 1
    double error2;          // angular displacement error of link 2
    double error2_velocity; // angular velocity error of link 2
    double r1;              // Eq. 2.9 define, filtered tracking error of link 1
    double r2;              // Eq. 2.9 define, filtered tracking error of link 2
    double Gamma[OUT][OUT];
    double dqr1;  // derivative of r1
    double dqr2;  // derivative of r2
    double ddqr1; // second-order derivative of r1
    double ddqr2; // second-order derivative of r2
    double Kv1;   // gain of link 1
    double Kv2;   // gain of link 2
    double F1;
    double F2;
    double k;
    double f1_estimate; // estimate of neural network function f
    double f2_estimate;
    double f_estimate_norm; // two-paradigm number of estimate of neural network function f
    double v1; // robust term
    double v2;
} controller;

void CONTROLLER_init(){
    system_state.q1 = 0.09;
    system_state.dq1 = 0.0;
    system_state.q2 = -0.90;
    system_state.dq2 = 0.0;
    controller.controller_u1 = q1_desired.y[0];
    controller.controller_u2 = dq1_desired.y[0];
    controller.controller_u3 = ddq1_desired.y[0];
    controller.controller_u4 = q2_desired.y[0];
    controller.controller_u6 = ddq2_desired.y[0];
    controller.controller_u7 = system_state.q1;
    controller.controller_u8 = system_state.dq1;
    controller.controller_u9 = system_state.q2;
    controller.controller_u10 = system_state.dq2;
    controller.error1 = q1_desired.y[0] - system_state.q1;
    controller.error1_velocity = dq1_desired.y[0] - system_state.dq1;
    controller.error2 = q2_desired.y[0] - system_state.q2;
    controller.error2_velocity = dq2_desired.y[0] - system_state.dq2;

    for (int j = 0; j < OUT; j++){
        for (int k = 0; k < OUT; k++){
            if (j == k){
                controller.Gamma[j][k] = 5.0;
            }
            else{
                controller.Gamma[j][k] = 0.0;
            }
        }
    }

    controller.r1 = controller.error1_velocity + controller.Gamma[0][0] * controller.error1; // filtered tracking error of link 1, r1, Eq. 2.9
    controller.r2 = controller.error2_velocity + controller.Gamma[1][1] * controller.error2; // filtered tracking error of link 2, r2, Eq. 2.9
    controller.Kv1 = 30;
    controller.Kv2 = 30;
    controller.F1 = 50;
    controller.F2 = 50;
}

struct _plant{
    double plant_u1;    
    double plant_u2;
    double plant_out1;
    double plant_out2;
    double plant_out3;
    double plant_out4;
    double plant_out5;
    double f[OUT]; // neural network function f
    double f_norm; // two-paradigm number of f
} plant;

void PLANT_init(){
    system_state.q1 = 0.09;
    system_state.dq1 = 0.0;
    system_state.q2 = -0.90;
    system_state.dq2 = 0.0;
    plant.plant_u1 = 0.0;
    plant.plant_u2 = 0.0;
    plant.plant_out1 = system_state.q1;
    plant.plant_out2 = system_state.dq1;
    plant.plant_out3 = system_state.q2;
    plant.plant_out4 = system_state.dq2;
}

double PLANT_realize(int i){
    plant.plant_u1 = torque.tol1;
    plant.plant_u2 = torque.tol2;
    dynamics.M[0][0] = p[0] + p[1] + 2 * p[2] * cos(system_state.q2);
    dynamics.M[0][1] = p[1] + p[2] * cos(system_state.q2);
    dynamics.M[1][0] = p[1] + p[2] * cos(system_state.q2);
    dynamics.M[1][1] = p[1];

    dynamics.Vm[0][0] = - p[2] * system_state.dq2 * sin(system_state.q2);
    dynamics.Vm[0][1] = - p[2] * (system_state.dq1 + system_state.dq2) * system_state.q2;
    dynamics.Vm[1][0] = p[2] * system_state.dq1 * sin(system_state.q2);
    dynamics.Vm[1][1] = 0;

    dynamics.G[0] = p[3] * g * cos(system_state.q1) + p[4] * g * cos(system_state.q1 + system_state.q2);
    dynamics.G[1] = p[4] * g * cos(system_state.q1 + system_state.q2);
    // printf("dynamics.G[1] = %f\n", dynamics.G[1]);

    if (system_state.dq1 > 0) {
        dynamics.F[0] = 0.2;
    } else if (system_state.dq1 < 0) {
        dynamics.F[0] = -0.2;
    }

    if (system_state.dq2 > 0) {
        dynamics.F[1] = 0.2;
    } else if (system_state.dq2 < 0) {
        dynamics.F[1] = -0.2;
    }

    torque.tol1_d = sin(i * Ts + t0);
    torque.tol2_d = sin(i * Ts + t0);

    double inv_M[OUT][OUT]; // inverse of inertia matrix
    inv_matrix(inv_M, dynamics.M, 2);

    double to1_Vmdq_G_F_told1, to1_Vmdq_G_F_told2;
    to1_Vmdq_G_F_told1 = torque.tol1 - (dynamics.Vm[0][0] * system_state.dq1 + dynamics.Vm[0][1] * system_state.dq2) - dynamics.G[0] - dynamics.F[0] - torque.tol1_d;
    to1_Vmdq_G_F_told2 = torque.tol2 - (dynamics.Vm[1][0] * system_state.dq1 + dynamics.Vm[1][1] * system_state.dq2) - dynamics.G[1] - dynamics.F[1] - torque.tol2_d;

    system_state.ddq1 = inv_M[0][0] * to1_Vmdq_G_F_told1 + inv_M[0][1] * to1_Vmdq_G_F_told2; // manipulator Lagrangian dynamics, Eq. 2.7
    system_state.ddq2 = inv_M[1][0] * to1_Vmdq_G_F_told1 + inv_M[1][1] * to1_Vmdq_G_F_told2;
    system_state.dq1 = system_state.dq1 + system_state.ddq1 * Ts;
    system_state.dq2 = system_state.dq2 + system_state.ddq2 * Ts;
    system_state.q1 = system_state.q1 + system_state.dq1 * Ts;
    system_state.q2 = system_state.q2 + system_state.dq2 * Ts;

    for (int j = 0; j < H; j++){
        plant.f[j] = dynamics.M[j][0] * controller.ddqr1 + dynamics.M[j][1] * controller.ddqr2 + dynamics.Vm[j][0] * controller.dqr1 + dynamics.Vm[j][1] * controller.dqr2 + dynamics.G[0] + dynamics.F[0]; // non-linear manipulator function, Eq. 2.11
    }

    plant.f_norm = sqrt(pow(plant.f[0],2) + pow(plant.f[1],2));
    archive.f_norm_archive[i] = plant.f_norm;

    plant.plant_out1 = system_state.q1;
    plant.plant_out2 = system_state.dq1;
    plant.plant_out3 = system_state.q2;
    plant.plant_out4 = system_state.dq2;
    plant.plant_out5 = plant.f_norm;

}

double CONTROL_realize(int i){
    controller.controller_u1 = q1_desired.y[i];
    controller.controller_u2 = dq1_desired.y[i];
    controller.controller_u3 = ddq1_desired.y[i];
    controller.controller_u4 = q2_desired.y[i];
    controller.controller_u6 = ddq2_desired.y[i];
    controller.controller_u7 = system_state.q1;
    controller.controller_u8 = system_state.dq1;
    controller.controller_u9 = system_state.q2;
    controller.controller_u10 = system_state.dq2;
    // printf("system_state.q1 = %f\n", system_state.q1);
    archive.q1_archive[i] = controller.controller_u7;
    archive.dq1_archive[i] = controller.controller_u8;
    archive.q2_archive[i] = controller.controller_u9;
    archive.dq2_archive[i] = controller.controller_u10;

    controller.error1 = q1_desired.y[i] - system_state.q1;
    controller.error1_velocity = dq1_desired.y[i] - system_state.dq1;
    controller.error2 = q2_desired.y[i] - system_state.q2;
    controller.error2_velocity = dq2_desired.y[i] - system_state.dq2;
    // printf("controller.err1 = %f\n", controller.err1);
    archive.error1_archive[i] = controller.error1;
    archive.error1_velocity_archive[i] = controller.error1_velocity;
    archive.error2_archive[i] = controller.error2;
    archive.error2_velocity_archive[i] = controller.error2_velocity;

    controller.r1 = controller.error1_velocity + controller.Gamma[0][0] * controller.error1; // filtered tracking error, Eq. 2.9
    controller.dqr1 = dq1_desired.y[i] + controller.Gamma[0][0] * controller.error1;
    controller.ddqr1 = ddq1_desired.y[i] + controller.Gamma[0][0] * controller.error1_velocity;
    controller.r2 = controller.error2_velocity + controller.Gamma[1][1] * controller.error2;
    controller.dqr2 = dq2_desired.y[i] + controller.Gamma[1][1] * controller.error2;
    controller.ddqr2 = dq2_desired.y[i] + controller.Gamma[1][1] * controller.error2_velocity;
    // printf("controller.r1 = %f\n", controller.r1);

    for (int j = 0; j < H; j++){
        phi1[j] = exp((-pow(controller.error1 - center[0][j], 2) - pow(controller.error1_velocity - center[1][j], 2) - pow(controller.controller_u1 - center[2][j], 2) 
            - pow(controller.controller_u2 - center[3][j], 2) - pow(controller.controller_u3 - center[4][j], 2)) / (width * width)); // output of RBF function
        // printf("phi1[%d] = %f\n", j, phi1[j]);
    }
    for (int j = 0; j < H; j++){
        phi2[j] = exp((-pow(controller.error2 - center[0][j], 2) - pow(controller.error2_velocity - center[1][j], 2) - pow(controller.controller_u4 - center[2][j], 2) 
            - pow(controller.controller_u5 - center[3][j], 2) - pow(controller.controller_u6 - center[4][j], 2)) / (width * width)); // output of RBF function
    }

    for (int j = 0; j < H; j++){
        weight1[j] = 0.0; // RBF network weight
        weight2[j] = 0.0;
    }

    // adaptive law
    if (CONTROLLAW == 1 || CONTROLLAW == 3 || CONTROLLAW == 4){

        for (int j = 0; j < H; j++){
            derivative_weight1[j] = controller.F1 * phi1[j] * controller.r1; // Eq. 3.12, derivative of weight estimate is equal to a constant F multiplied by output of hidden layer of RBF network multiplied by filtered tracking error r
            // printf("derivative_weight1[%d] = %f\n", j, derivative_weight1[j]);
        }

        for (int j = 0; j < H; j++){
            derivative_weight2[j] = controller.F2 * phi2[j] * controller.r2;
        }

    }
    else if (CONTROLLAW == 2){

        for (int j = 0; j < H; j++){
            derivative_weight1[j] = controller.F1 * phi1[j] * controller.r1 - controller.k * controller.F1 * sqrt(pow(controller.r1, 2) + pow(controller.r2, 2)) * weight1[j];
            // Eq. 3.21, derivative of weight estimate is equal to product of a constant F multiplied by output of hidden layer of RBF network multiplied by filtered tracking error r minus the product of design parameter k multiplied by F multiplied by L2 norm of r multiplied by weight estimate w
        }

        for (int j = 0; j < H; j++){
            derivative_weight2[j] = controller.F2 * phi2[j] * controller.r2 - controller.k * controller.F2 * sqrt(pow(controller.r1, 2) + pow(controller.r2, 2)) * weight2[j];
        }

    }

    for (int j = 0; j < H; j++){
        weight1[j] = weight1[j] + derivative_weight1[j] * Ts;
    }
    for (int j = 0; j < H; j++){
        weight2[j] = weight2[j] + derivative_weight2[j] * Ts;
    }

    controller.f1_estimate = 0.0;
    for (int j = 0; j < H; j++){
        controller.f1_estimate += weight1[j] * phi1[j]; // E.q 3.4, estimate of neural network function f is equal to estimate of weight multiplied by hidden layer output
    }

    controller.f2_estimate = 0.0;
    for (int j = 0; j < H; j++){
        controller.f2_estimate += weight2[j] * phi2[j];
    }

    // control law
    if (CONTROLLAW == 1 || CONTROLLAW == 2){
        torque.tol1 = controller.f1_estimate + controller.Kv1 * controller.r1; // Eq. 2.13 control input torque tol0 equals estimate of function f(x) plus product of gain matrix kv times filtered tracking error r
        torque.tol2 = controller.f2_estimate + controller.Kv2 * controller.r2;
    }
    else if (CONTROLLAW == 3){
        double epsilon_N = 0.20;
        double bd = 1;
        double gamma = 0.5;
        double sat_r1, sat_r2;

        if (controller.r1 >= 0){
            sat_r1 = 1 - exp(-controller.r1 / gamma); // sat(r) equals 1 minus the xth power of e, x equals negative r divided by gamma
        }
        else{
            sat_r1 = -(1 - exp(controller.r1 / gamma));
        }

        if (controller.r2 >= 0){
            sat_r2 = 1 - exp(-controller.r2 / gamma);
        }
        else{
            sat_r2 = -(1 - exp(controller.r2 / gamma));
        }

        controller.v1 = -(epsilon_N + bd) * sat_r1; // Eq. 3.26 robust term v equals - ( bounding function epsilon_N + upper limit of control torque) times saturation function sat(r)
        controller.v2 = -(epsilon_N + bd) * sat_r2;

        torque.tol1 = controller.f1_estimate + controller.Kv1 * controller.r1 - controller.v1; // Eq. 3.7 control torque tol equals control input torque tol0 minus robust term v
        torque.tol2 = controller.f2_estimate + controller.Kv2 * controller.r2 - controller.v2;
    }
    else if (CONTROLLAW == 4){
        double epsilon_N = 0.20;
        double bd = 1;

        if (controller.r1 >= 0){
            controller.v1 = -(epsilon_N + bd);
        }
        else{
            controller.v1 = epsilon_N + bd;
        }

        if (controller.r2 >= 0){
            controller.v2 = -(epsilon_N + bd);
        }
        else{
            controller.v2 = epsilon_N + bd;
        }

        torque.tol1 = controller.f1_estimate + controller.Kv1 * controller.r1 - controller.v1;
        torque.tol2 = controller.f2_estimate + controller.Kv2 * controller.r2 - controller.v2;
    }
    else if (CONTROLLAW == 5){ // only PD control, with no neural network
        torque.tol1 = controller.f1_estimate + controller.Kv1 * controller.r1;
        torque.tol2 = controller.f2_estimate + controller.Kv2 * controller.r2;
    }

    archive.tol1_archive[i] = torque.tol1;
    archive.tol2_archive[i] = torque.tol2;

    controller.f_estimate_norm = sqrt(pow(controller.f1_estimate, 2) + pow(controller.f2_estimate, 2));
    archive.f_estimate_norm_archive[i] = controller.f_estimate_norm;
    //printf("f_estimate_norm: %.2f\n", controller.f_estimate_norm);

    controller.controller_out1 = torque.tol1;
    controller.controller_out2 = torque.tol2;
    controller.controller_out3 = controller.f_estimate_norm;
}

void saveArchiveToTxt(double *archive, int size, const char *filename){

    FILE *file = fopen(filename, "w+");

    if (file == NULL){
        perror("Failed to open file");
        exit(1);
    }
    else{
        for (int i = 0; i < size; i++){
            fprintf(file, "%lf\n", archive[i]);
        }
        fclose(file);
        printf("Saved to file %s\n", filename);
    }
}

void saveArchive(){

        saveArchiveToTxt(q1_desired.y, ARRAY_SIZE, "../report/qd1.txt");
        saveArchiveToTxt(archive.q1_archive, ARRAY_SIZE, "../report/q1.txt");
        saveArchiveToTxt(archive.dq1_archive, ARRAY_SIZE, "../report/dq1.txt");
        saveArchiveToTxt(q2_desired.y, ARRAY_SIZE, "../report/qd2.txt");
        saveArchiveToTxt(archive.q2_archive, ARRAY_SIZE, "../report/q2.txt");
        saveArchiveToTxt(archive.dq2_archive, ARRAY_SIZE, "../report/dq2.txt");
        saveArchiveToTxt(archive.error1_archive, ARRAY_SIZE, "../report/error1.txt");
        saveArchiveToTxt(archive.error1_velocity_archive, ARRAY_SIZE, "../report/error1_velocity.txt");
        saveArchiveToTxt(archive.error2_archive, ARRAY_SIZE, "../report/error2.txt");
        saveArchiveToTxt(archive.error2_velocity_archive, ARRAY_SIZE, "../report/error2_velocity.txt");
        saveArchiveToTxt(archive.tol1_archive, ARRAY_SIZE, "../report/tol1.txt");
        saveArchiveToTxt(archive.tol2_archive, ARRAY_SIZE, "../report/tol2.txt");
        saveArchiveToTxt(archive.f_estimate_norm_archive, ARRAY_SIZE, "../report/f_estimate_norm_archive.txt");
        saveArchiveToTxt(archive.f_norm_archive, ARRAY_SIZE, "../report/f_norm_archive.txt");

}

int main(){

    SystemInput(&q1_desired, &dq1_desired, &ddq1_desired, &q2_desired, &dq2_desired, &ddq2_desired, Ts, t0, t1);
    CONTROLLER_init(); // initialize controller parameter
    PLANT_init();   // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++){
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        CONTROL_realize(i);
        PLANT_realize(i);
    }

    saveArchive();

    return 0;
}
