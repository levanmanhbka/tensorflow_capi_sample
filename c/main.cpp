/*
manh.lv run on:
Visual Studio Professional 2019
Cuda 10.1
Tensorflow 2.31
Tensorflow CAPI: https://www.tensorflow.org/install/lang_c
CAPI Version: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-2.3.1.zip
*/
#include<iostream>
using namespace std;

extern "C" {
#include <stdlib.h>
#include <stdio.h>
#include<tensorflow/c/c_api.h>
}

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main()
{
	cout << "Hello Tensorflow for C" << endl;
	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();

	TF_SessionOptions* session_opts = TF_NewSessionOptions();
	TF_Buffer* run_opts = NULL;

    /*
    The given SavedModel SignatureDef contains the following input(s):
      inputs['input_1'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 28, 28, 1)
          name: serving_default_input_1:0
    The given SavedModel SignatureDef contains the following output(s):
      outputs['dense_2'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 10)
          name: StatefulPartitionedCall:0
    Method name is: tensorflow/serving/predict
    */
    const char* saved_model_dir = "image_classification/";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* session = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph, NULL, status);
    if (TF_GetCode(status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s", TF_Message(status));
    }

    //****** Get input tensor
    int num_inputs = 1;
    TF_Output* input = (TF_Output*)malloc(sizeof(TF_Output) * num_inputs);

    TF_Output t0 = { TF_GraphOperationByName(graph, "serving_default_input_1"), 0 };
    if (t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

    input[0] = t0;


     //********* Get output tensor
    int num_outputs = 1;
    TF_Output* output = (TF_Output*)malloc(sizeof(TF_Output) * num_outputs);

    TF_Output t2 = { TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0 };
    if (t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
        printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    output[0] = t2;

    //********* Allocate data for inputs & outputs
    TF_Tensor** input_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * num_inputs);
    TF_Tensor** output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * num_outputs);

    int ndims = 4;
    int64_t dims[] = { 1, 28, 28, 1 };
    float data[1 * 28 * 28 * 1];
    for (int i = 0; i < (1 * 28 * 28 * 1); i++)
    {
        data[i] = 1.00;
    }
    int ndata = sizeof(float) * 1 * 28 * 28 * 1;// This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
        printf("TF_NewTensor is OK\n");
    }
    else
        printf("ERROR: Failed TF_NewTensor\n");

    input_values[0] = int_tensor;

    // //Run the session
    TF_SessionRun(session, NULL, input, input_values, num_inputs, output, output_values, num_outputs, NULL, 0, NULL, status);

    if (TF_GetCode(status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s", TF_Message(status));
    }

    void* buff = TF_TensorData(output_values[0]);
    float* offsets = (float*)buff;
    printf("Result Tensor :\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", offsets[i]);
    }
    
    //Free memory
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteStatus(status);

    return 0;
}