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
  int* weights;           // The weight of each connection
} TargetArray;

typedef struct {
  int sizeInputLayer;     // First n neurons receive external inputs
  int sizeOutputLayer;    // Final n neurons are outputs
  int* bias;              // bias[i]: bias of neuron i
  int* gain;              // gain[i]: gain of neuron i
  int* constInput;        // constInput[i]: constant input to neuron i
  TargetArray* targets;   // targets[i]: array of targets from neuron i
  TargetArray* inTargets; // inTargets[i]: array of targets from input i
} Network;

// Fixed-point multiply =======================================================

// Integer n represents fixed-point number n/1024.

inline int mul(int x, int y) { return (x*y)>>10; }

// IO =========================================================================

// Read neural network for handwriting recognition.

int** readMatrix(const int* data, int rows, int cols)
{
  int i, j;
  int** m;
  m = malloc(rows * sizeof(int*));
  for (i = 0; i < rows; i++)
    m[i] = (int*) &data[i*cols];
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
int sigmoid_bias[]    = {10567, 6092, 655};
int sigmoid_gain[]    = {8990, 1413, 10260};
int sigmoid_decoder[] = {1054, 1402, 880};

void createConstInput(int* constInput, int numNeuron, 
                        int lifPerNeuron, int** inp)
{
  int i, j;
  for (i = 0; i < numNeuron; i++) {
    int base = i*lifPerNeuron;
    for (j = 0; j < lifPerNeuron; j++)
      constInput[base+j] = inp[0][i];
  }
}

void createConnections(
    TargetArray* t
  , int numSource   , int numTarget
  , int lifPerSource, int lifPerTarget
  , int targetBase
  , int** weights
  )
{
  int i, j, x, y;

  for (i = 0; i < numSource; i++) {
    int base = i*lifPerSource;
    for (j = 0; j < lifPerSource; j++) {
      int space = numTarget*lifPerTarget;
      t[base+j].numTargets = 0;
      t[base+j].targets = malloc(sizeof(int) * space);
      t[base+j].weights = malloc(sizeof(int) * space);
      for (x = 0; x < numTarget; x++)
       if (weights[i][x] != 0)
        for (y = 0; y < lifPerTarget; y++) {
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
  int i, j;

  /* Read the 4-layer RBM network available from nengo.ca */
  int** b1 = readMatrix(mat_1_b, 1, 1000);
  int** w1 = readMatrix(mat_1_w, 784, 1000);
  int** b2 = readMatrix(mat_2_b, 1, 500);
  int** w2 = readMatrix(mat_2_w, 1000, 500);
  int** b3 = readMatrix(mat_3_b, 1, 300);
  int** w3 = readMatrix(mat_3_w, 500, 300);
  int** b4 = readMatrix(mat_4_b, 1, 50);
  int** w4 = readMatrix(mat_4_w, 300, 50);

  // Allocate Network
  Network* net = malloc(sizeof(Network));
  net->sizeInputLayer = 1000*LIF_PER_RBM;
  net->sizeOutputLayer = NUM_OUTPUTS;
  net->gain = malloc(sizeof(int) * NUM_LIF_NEURONS);
  net->bias = malloc(sizeof(int) * NUM_LIF_NEURONS);
  net->constInput = malloc(sizeof(int) * NUM_LIF_NEURONS);
  net->inTargets = malloc(sizeof(TargetArray) * NUM_INPUTS);
  net->targets = malloc(sizeof(TargetArray) * NUM_LIF_NEURONS);

  // Set gain and bias of each internal LIF neuron
  for (i = 0; i < NUM_RBM_NEURONS; i++) {
    int base = i*LIF_PER_RBM;
    for (j = 0; j < LIF_PER_RBM; j++) {
      net->gain[base+j] = sigmoid_gain[j];
      net->bias[base+j] = sigmoid_bias[j];
    }
  }

  // Layer 1
  createConstInput(net->constInput, 1000, LIF_PER_RBM, b1);
  createConnections(net->inTargets, 784, 1000, 1, LIF_PER_RBM, 0, w1);

  // Layer 2
  createConstInput(&net->constInput[1000*LIF_PER_RBM], 500, LIF_PER_RBM, b2);
  createConnections(net->targets, 1000, 500, LIF_PER_RBM, LIF_PER_RBM,
                      1000*LIF_PER_RBM, w2);

  // Layer 3
  createConstInput(&net->constInput[(1000+500)*LIF_PER_RBM],
                     300, LIF_PER_RBM, b3);
  createConnections(&net->targets[1000*LIF_PER_RBM], 500, 300,
                      LIF_PER_RBM, LIF_PER_RBM, (1000+500)*LIF_PER_RBM, w3);

  // Layer 4
  createConstInput(&net->constInput[(1000+500+300)*LIF_PER_RBM], 50, 1, b4);
  createConnections(&net->targets[(1000+500)*LIF_PER_RBM], 300, 50,
                      LIF_PER_RBM, 1, (1000+500+300)*LIF_PER_RBM, w4);

  // Output layer
  for (i = NUM_LIF_NEURONS-NUM_OUTPUTS; i < NUM_LIF_NEURONS; i++) {
    net->gain[i] = net->bias[i] = net->constInput[i] = 0;
    net->targets[i].numTargets = 0;
  }

  return net;
}

int dot(int* v, int* w)
{
  int i;
  int acc = 0;
  for (i = 0; i < 50; i++)
    acc += mul(v[i],w[i]);
  return acc;
}

// For each i in 0..9, ans[i] is updated to contain a "likeness" score
// in the range 0 to 140.
void answer(Network* net, int** semPtr, int* inp, int* ans)
{
  int* out = inp+(NUM_LIF_NEURONS-net->sizeOutputLayer);
  int maxScore = 0x80000000;
  int minScore = 0x7fffffff;
  int i;
    
  for (i = 0; i < 10; i++) {
    ans[i] = dot(semPtr[i], out);
    if (ans[i] >= maxScore) maxScore = ans[i];
    if (ans[i] <= minScore) minScore = ans[i];
  }

  // Make scores start from 0
  maxScore += -minScore;
  for (i = 0; i < 10; i++)
    ans[i] += -minScore;

  // Put in range 0..160
  for (i = 0; i < 10; i++)
    ans[i] = (ans[i]*140) / maxScore;
}

// LIF simulator ==============================================================

const int one_over_rc = 50;  // 1/t_rc
const int t_ref       = 2;   // Milliseconds
const int pstc_scale  = 158; // 1-e^(-dt/t_pstc);

int runNeurons(int* input, int* v, int* ref, int* spikes)
{
  int i;
  int numSpikes = 0;

  for (i = 0; i < NUM_LIF_NEURONS; i++) {
    int dV = mul(input[i]-v[i],
                 one_over_rc);            // the LIF voltage change equation
    v[i] += dV;
    if (v[i] < 0) v[i] = 0;               // don't allow voltage to go below 0

    if (ref[i] > 0) {                     // if we are in our refractory period
      v[i] = 0;                           //   keep voltage at zero and
      ref[i] -= 1;                        //   decrease the refractory period
    }

    if (v[i] > 1024) {                    // if we have hit threshold
      spikes[numSpikes++] = i;            //   spike
      v[i] = 0;                           //   reset the voltage
      ref[i] = t_ref;                     //   and set the refractory period
    }
  }
  return numSpikes;
}

void simulate(
    Network* net
  , int* v          // Voltage of each LIF neuron
  , int* ref        // Refactory period of each LIF neuron
  , int* inp        // Current input to each neuron
  , int* total      // Input to each neuron (after applying gain and bias)
  , int ms          // Number of milliseconds to simulate
  , int* spikes
  , int* spikeCount
  )
{
  int numSpikes = 0;
  int i, j, t;

  for (t = 0; t < ms; t++) {
    // Decay neuron inputs (implementing the post-synaptic filter)
    // except input layer
    for (i = net->sizeInputLayer; i < NUM_LIF_NEURONS; i++)
      inp[i] = mul(inp[i], 1024-pstc_scale);

    // For each neuron that spikes, increase the input current
    // of all the neurons it is connected to by the synaptic
    // connection weight
    for (i = 0; i < numSpikes; i++) {
      TargetArray t = net->targets[spikes[i]];
      spikeCount[spikes[i]]++;
      for (j = 0; j < t.numTargets; j++)
        inp[t.targets[j]] += mul(t.weights[j],pstc_scale);
    }

    // Compute the total input into each neuron
    for (i = 0; i < NUM_LIF_NEURONS; i++)
      total[i] = inp[i];

    // Assign constant input
    for (i = 0; i < NUM_LIF_NEURONS; i++)
      total[i] += net->constInput[i];
    // Apply gain and bias
    for (i = 0; i < NUM_LIF_NEURONS; i++)
      total[i] = mul(net->gain[i],total[i])+net->bias[i];

    numSpikes = runNeurons(total, v, ref, spikes);
  }
}

// Digit recognition interface ================================================

typedef struct {
  Network* net;
  int* v;          // Voltage of each neuron
  int* ref;        // Refactory period of each neuron
  int* inp;        // Input to each neuron
  int* total;      // Input after apply gain and bias
  int* spikes;     // Spike buffer
  int** semPtr;    // Used to map neural net output to digit
  int** samples;   // Sample images
  int* spikeCount; // Spike count (for drawing a spike frequency plot)
} Recogniser;

void assignExternalInput(Network* net, int* total, int* externalInput)
{
  int i, j;

  for (i = 0; i < net->sizeInputLayer; i++) total[i] = 0;

  for (i = 0; i < NUM_INPUTS; i++) {
    TargetArray t = net->inTargets[i];
    for (j = 0; j < t.numTargets; j++)
      total[t.targets[j]] += mul(t.weights[j], externalInput[i]);
  }
}

Recogniser* createRecogniser()
{
  Recogniser* r = malloc(sizeof(Recogniser));
  r->semPtr = readMatrix(SemPtr, 10, 50);
  r->net = createNetwork();
  r->v = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->ref = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->inp = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->total = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->spikeCount = calloc(NUM_LIF_NEURONS, sizeof(int));
  r->spikes = malloc(sizeof(int) * NUM_LIF_NEURONS);
  r->samples = readMatrix(samplesPtr, 100, 784);
  return r;
}

void recognise(Recogniser* r, int *image28by28, int* ans)
{
  memset(r->v, 0, sizeof(int)*NUM_LIF_NEURONS);
  memset(r->ref, 0, sizeof(int)*NUM_LIF_NEURONS);
  memset(r->inp, 0, sizeof(int)*NUM_LIF_NEURONS);
  memset(r->total, 0, sizeof(int)*NUM_LIF_NEURONS);
  memset(r->spikeCount, 0, sizeof(int)*NUM_LIF_NEURONS);
  assignExternalInput(r->net, r->inp, image28by28);
  simulate(r->net, r->v, r->ref, r->inp, r->total, 20, r->spikes, r->spikeCount);
  answer(r->net, r->semPtr, r->inp, ans);
}

int best(int* ans)
{
  int i, b;
  int max = 0x80000000;
  for (i = 0; i < 10; i++) {
    if (ans[i] > max) { max = ans[i]; b = i; }
  }
  return b;
}


void main()
{
  int i;
  int answer[10];
  Recogniser* r = createRecogniser();
  // Do recognition on 100 samples
  // The sample set contains 10 of each digit, sorted by digit.
  //
  //
  //
  int  expected_ans[]  =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 9, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  for (i = 0; i < 100; i++) {
    recognise(r, r->samples[i], answer);
    int ans = best(answer);
    printf("%i ", ans);

    assert( ans == expected_ans[i] );
  }
  printf("\n");
}
