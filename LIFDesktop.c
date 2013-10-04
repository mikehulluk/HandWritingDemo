// An LIF neuron simulator in C
// with an example network for handwritten-digit recognition
// based on work by Terrence C. Stewart

// PC desktop version.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "Handwriting.h"

// Data structures ============================================================

typedef struct {
  int numTargets;         // The number of neurons we are connected to
  int* targets;           // The neuron ids we are connected to
  float* weights;           // The weight of each connection
} TargetArray;

typedef struct {
  int sizeInputLayer;     // First n neurons receive external inputs
  int sizeOutputLayer;    // Final n neurons are outputs
  float* bias;              // bias[i]: bias of neuron i
  float* gain;              // gain[i]: gain of neuron i
  float* constInput;        // constInput[i]: constant input to neuron i
  TargetArray* targets;   // targets[i]: array of targets from neuron i
  TargetArray* inTargets; // inTargets[i]: array of targets from input i
} Network;

// IO =========================================================================

// Read neural network for handwriting recognition.

float** readMatrix(const float* data, int rows, int cols)
{
  float** m;
  m = malloc(rows * sizeof(float*));
  for (int i = 0; i < rows; i++)
    m[i] = (float*) &data[i*cols];
  return m;
}

float** readFloatMatrix(const float* data, int rows, int cols, float downscale)
{
  float** m;
  m = malloc(rows * sizeof(float*));
  for (int i = 0; i < rows; i++)
  {
    m[i] = malloc( cols * sizeof(float) );
    memcpy(m[i],  &data[i*cols], sizeof(float)*cols );

    for(int x=0;x<cols;x++)
    {
        m[i][x] /= downscale;
    }
  }
  return m;
}




// Sample network (performs hand-written digit recognition) ===================

// Each linear RBM neuron is represented by a trio of spiking LIF neurons
#define LIF_PER_RBM 3

#define NUM_INPUTS      784
#define NUM_OUTPUTS     50
#define NUM_RBM_NEURONS (1000+500+300)
#define NUM_LIF_NEURONS (NUM_RBM_NEURONS*LIF_PER_RBM + NUM_OUTPUTS)

// The bias, gain and decoder of every LIF neuron trio
//int sigmoid_bias[]    = {10567, 6092, 655};
//int sigmoid_gain[]    = {8990, 1413, 10260};
//int sigmoid_decoder[] = {1054, 1402, 880};

float sigmoid_bias_fl[]    = {10.31933, 5.94921, 0.63964};
float sigmoid_gain_fl[]    = {8.7792, 1.37988, 10.0195};
float sigmoid_decoder_fl[] = {1.0292, 1.36914, 0.85937};

void createConstInput(float* constInput, int numNeuron,
                        int lifPerNeuron, float** inp)
{
  for (int i = 0; i < numNeuron; i++) {
    int base = i*lifPerNeuron;
    for (int j = 0; j < lifPerNeuron; j++)
      constInput[base+j] = inp[0][i]; 
  }
}

void createConnections(
    TargetArray* t
  , int numSource   , int numTarget
  , int lifPerSource, int lifPerTarget
  , int targetBase
  , float** weights
  )
{

  for (int i = 0; i < numSource; i++) {
    int base = i*lifPerSource;
    for (int j = 0; j < lifPerSource; j++) {
      int space = numTarget*lifPerTarget;
      t[base+j].numTargets = 0;
      t[base+j].targets = malloc(sizeof(int) * space);
      t[base+j].weights = malloc(sizeof(float) * space);
      for (int x = 0; x < numTarget; x++)
       if (weights[i][x] != 0)
        for (int y = 0; y < lifPerTarget; y++) {
          int n = t[base+j].numTargets;
          t[base+j].targets[n] = targetBase + x*lifPerTarget + y;
          t[base+j].weights[n] = weights[i][x];
          t[base+j].numTargets++;
        }
    }
  }
}

Network* createNetwork()
{

  /* Read the 4-layer RBM network available from nengo.ca */
  float** b1 = readFloatMatrix(mat_1_b, 1, 1000,   1024);
  float** w1 = readFloatMatrix(mat_1_w, 784, 1000, 1);
  float** b2 = readFloatMatrix(mat_2_b, 1, 500,    1024);
  float** w2 = readFloatMatrix(mat_2_w, 1000, 500, 1);
  float** b3 = readFloatMatrix(mat_3_b, 1, 300,    1024);
  float** w3 = readFloatMatrix(mat_3_w, 500, 300,  1);
  float** b4 = readFloatMatrix(mat_4_b, 1, 50,     1024);
  float** w4 = readFloatMatrix(mat_4_w, 300, 50,   1);

  // Allocate Network
  Network* net = malloc(sizeof(Network));
  net->sizeInputLayer = 1000*LIF_PER_RBM;
  net->sizeOutputLayer = NUM_OUTPUTS;
  net->gain = malloc(sizeof(float) * NUM_LIF_NEURONS);
  net->bias = malloc(sizeof(float) * NUM_LIF_NEURONS);
  net->constInput = malloc(sizeof(float) * NUM_LIF_NEURONS);
  net->inTargets = malloc(sizeof(TargetArray) * NUM_INPUTS);
  net->targets = malloc(sizeof(TargetArray) * NUM_LIF_NEURONS);

  // Set gain and bias of each internal LIF neuron
  for (int i = 0; i < NUM_RBM_NEURONS; i++) {
    int base = i*LIF_PER_RBM;
    for (int j = 0; j < LIF_PER_RBM; j++) {
      net->gain[base+j] = sigmoid_gain_fl[j];
      net->bias[base+j] = sigmoid_bias_fl[j];
    }
  }

  // Layer 1
  createConstInput(net->constInput, 1000, LIF_PER_RBM, b1);
  createConnections(net->inTargets, 784, 1000, 1, LIF_PER_RBM, 0, w1);

  // Layer 2
  createConstInput(&net->constInput[1000*LIF_PER_RBM], 500, LIF_PER_RBM, b2);
  createConnections(net->targets, 1000, 500, LIF_PER_RBM, LIF_PER_RBM, 1000*LIF_PER_RBM, w2);

  // Layer 3
  createConstInput(&net->constInput[(1000+500)*LIF_PER_RBM], 300, LIF_PER_RBM, b3);
  createConnections(&net->targets[1000*LIF_PER_RBM], 500, 300, LIF_PER_RBM, LIF_PER_RBM, (1000+500)*LIF_PER_RBM, w3);

  // Layer 4
  createConstInput(&net->constInput[(1000+500+300)*LIF_PER_RBM], 50, 1, b4);
  createConnections(&net->targets[(1000+500)*LIF_PER_RBM], 300, 50, LIF_PER_RBM, 1, (1000+500+300)*LIF_PER_RBM, w4);

  // Output layer
  for (int i = NUM_LIF_NEURONS-NUM_OUTPUTS; i < NUM_LIF_NEURONS; i++) {
    net->gain[i] = net->bias[i] = net->constInput[i] = 0;
    net->targets[i].numTargets = 0;
  }

  return net;
}

float dot(float* v, float* w)
{
  float acc = 0;
  for (int i = 0; i < 50; i++)
    acc += (v[i]/1024) * w[i];
  return acc;
}

// For each i in 0..9, ans[i] is updated to contain a "likeness" score
// in the range 0 to 140.
void answer(Network* net, float** semPtr, float* inp, int* ans)
{

  float* out = inp +(NUM_LIF_NEURONS-net->sizeOutputLayer);
  int maxScore = 0x80000000;
  int minScore = 0x7fffffff;


  for (int i = 0; i < 50; i++) {
        //printf("\nOut: %d %f", i, out[i]);
  }
  //return; 


  for (int i = 0; i < 10; i++) {
    ans[i] = dot(semPtr[i], out);
    if (ans[i] >= maxScore) maxScore = ans[i];
    if (ans[i] <= minScore) minScore = ans[i];
  }

  // Make scores start from 0
  maxScore += -minScore;
  for (int i = 0; i < 10; i++)
    ans[i] += -minScore;

  // Put in range 0..160
  for (int i = 0; i < 10; i++)
    ans[i] = (ans[i]*140) / maxScore;
}

// LIF simulator ==============================================================

const int one_over_rc = 50;  // 1/t_rc
const int t_ref       = 2;   // Milliseconds
const int pstc_scale  = 158; // 1-e^(-dt/t_pstc);


#define one_over_rc_float ((one_over_rc)/1024.)
#define pstc_scale_float  ((pstc_scale) / 1024.)


int runNeurons(float* input, float* v, float* ref, int* spikes)
{

  int numSpikes = 0;

  for (int i = 0; i < NUM_LIF_NEURONS; i++) {
    // the LIF voltage change equation
    v[i] += (input[i]-v[i]) * one_over_rc_float;

    if (v[i] < 0) v[i] = 0;               // don't allow voltage to go below 0

    if (ref[i] > 0) {                     // if we are in our refractory period
      v[i] = 0;                           //   keep voltage at zero and
      ref[i] -= 1;                        //   decrease the refractory period
    }

    if (v[i] > 1.0) {                    // if we have hit threshold
      spikes[numSpikes++] = i;            //   spike
      v[i] = 0;                           //   reset the voltage
      ref[i] = t_ref;                     //   and set the refractory period
    }
  }
  return numSpikes;
}




void simulate(
    Network* net
  , float* v          // Voltage of each LIF neuron
  , float* ref        // Refactory period of each LIF neuron
  , float* inp        // Current input to each neuron
  , float* total      // Input to each neuron (after applying gain and bias)
  , int ms          // Number of milliseconds to simulate
  , int* spikes
  , int* spikeCount
  )
{
  int numSpikes = 0;

  for (int t = 0; t < ms; t++) {

    // For each neuron that spikes, increase the input current
    // of all the neurons it is connected to by the synaptic
    // connection weight
    for (int i = 0; i < numSpikes; i++) {
      TargetArray t = net->targets[spikes[i]];
      spikeCount[spikes[i]]++;
      for (int j = 0; j < t.numTargets; j++)
      {
        inp[t.targets[j]] += (t.weights[j] * pstc_scale_float) /1024.; 
      }
    }


    // Compute the total input into each neuron
    for (int i = 0; i < NUM_LIF_NEURONS; i++)
    {
      float s = (inp[i] + net->constInput[i]);
      total[i] = (net->gain[i] * s )+net->bias[i];
    }


    numSpikes = runNeurons(total, v, ref, spikes);

    // Decay neuron inputs (implementing the post-synaptic filter)
    // except input layer
    for (int i = net->sizeInputLayer; i < NUM_LIF_NEURONS; i++)
    {
      inp[i] *= (1.- pstc_scale_float) ;
    }
  }




    for (int i = 0; i < NUM_LIF_NEURONS; i++)
    {
      inp[i] *= 1024;
    }


}

// Digit recognition interface ================================================

typedef struct {
  Network* net;
  float* v;          // Voltage of each neuron
  float* ref;        // Refactory period of each neuron
  float* inp;        // Input to each neuron
  float* total;      // Input after apply gain and bias
  int* spikes;     // Spike buffer
  float** semPtr;    // Used to map neural net output to digit
  float** samples;   // Sample images
  int* spikeCount; // Spike count (for drawing a spike frequency plot)
} Recogniser;

void assignExternalInput(Network* net, float* total, float* externalInput)
{
  int i, j;

  for (i = 0; i < net->sizeInputLayer; i++) total[i] = 0;

  for (i = 0; i < NUM_INPUTS; i++) {
    TargetArray t = net->inTargets[i];
    for (j = 0; j < t.numTargets; j++)
      total[t.targets[j]] += (t.weights[j] * externalInput[i] ) / 1024. / 1024.;
  }
}

Recogniser* createRecogniser()
{
  Recogniser* r = malloc(sizeof(Recogniser));
  r->semPtr = readMatrix(SemPtr, 10, 50);
  r->net = createNetwork();
  r->v = calloc(NUM_LIF_NEURONS, sizeof(float));
  r->ref = calloc(NUM_LIF_NEURONS, sizeof(float));
  r->inp = calloc(NUM_LIF_NEURONS, sizeof(float));
  r->total = calloc(NUM_LIF_NEURONS, sizeof(float));
  r->spikeCount = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->spikes = malloc(sizeof(int) * NUM_LIF_NEURONS);
  r->samples = readMatrix(samplesPtr, 100, 784);
  return r;
}

void recognise(Recogniser* r, float *image28by28, int* ans)
{
  memset(r->v, 0, sizeof(float)*NUM_LIF_NEURONS);
  memset(r->ref, 0, sizeof(float)*NUM_LIF_NEURONS);
  memset(r->inp, 0, sizeof(float)*NUM_LIF_NEURONS);
  memset(r->total, 0, sizeof(float)*NUM_LIF_NEURONS);
  memset(r->spikeCount, 0, sizeof(int)*NUM_LIF_NEURONS);
  assignExternalInput(r->net, r->inp, image28by28);
  simulate(r->net, r->v, r->ref, r->inp, r->total, 20, r->spikes, r->spikeCount);
  answer(r->net, r->semPtr, r->inp, ans);
}

int best(int* ans)
{
  int b = 0;
  int max = 0x80000000;
  for (int i = 0; i < 10; i++) {
    if (ans[i] > max) { max = ans[i]; b = i; }
  }
  return b;
}


int main()
{
  int i;
  int answer[10];
  Recogniser* r = createRecogniser();
  // Do recognition on 100 samples
  // The sample set contains 10 of each digit, sorted by digit.
  //
  int expected_ans[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  for (i = 0; i < 100; i++) {
    recognise(r, r->samples[i], answer);
    int ans = best(answer);
    printf("%i ", ans);

    assert( ans == expected_ans[i] );
  }
  printf("\n");

  return 0;
}
