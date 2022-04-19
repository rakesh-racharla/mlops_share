# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy


sess = rt.InferenceSession("C:\\Users\\rakes\\PycharmProjects\\MLops_project_one\\mlruns\\0\\1bc00dae97954f2a861c9e4da877b48d\\artifacts\\model\\model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
inf_data = numpy.array([1.8,9.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8])
pred_onx = sess.run([label_name], {input_name: numpy.expand_dims(inf_data.astype(numpy.float32),axis=0)})[0]
pred_onx

pred_onx[0][0]


import mlflow

mlflow.list_experiments()

mlflow.active_run()
mlflow.search_runs(experiment_name=['Default'])
mlflow.get_experiment('0')
mlflow.search_runs(experiment_ids=["0"])

from mlflow.tracking.client import MlflowClient
MlflowClient().list_experiments()

from mlflow.entities import ViewType
MlflowClient().search_runs(experiment_ids=["0"], run_view_type=ViewType.ACTIVE_ONLY)



mlflow.get_run()
