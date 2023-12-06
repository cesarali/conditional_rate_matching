from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import ReferenceProcess

def check_model_devices(x):
    if isinstance(x,ReferenceProcess):
        return x.device
    else:
        return x.parameters().__next__().device