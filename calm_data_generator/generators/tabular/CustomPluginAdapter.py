import pandas as pd 

class CustomPluginAdapter: 
    """Class that allow the use of any model of the libraries that we use """
    
    def __init__(self, model, fit_fn=None, generate_fn=None, postprocess_fn=None, columns=None, method_name:str ="custom"):
        self.model = model 
        self.columns = columns 
        self.method_name = method_name
        self._is_fitted = False
        
        self._fit_fn  = fit_fn or self._detect_fit_fn(model)
        self._generate_fn =  generate_fn or self._detect_generate_fn(model)
        self._postprocess_fn = postprocess_fn
        
    def _detect_fit_fn(self, model): 
        if hasattr(model, 'fit'):
            return lambda m, data: m.fit(data)
        elif hasattr(model, 'train'): 
            return lambda m, data: m.train(data)
        else: 
            raise ValueError("No fit method found. Pass fit_fn explicitly.")
    
    def _detect_generate_fn(self, model): 
        if hasattr(model, 'generate'): 
            return lambda m, n: m.generate(count=n).dataframe()      
        elif hasattr(model, 'sample'):
           return lambda m, n: pd.DataFrame(m.sample(n)[0], columns=self.columns)
        elif hasattr(model, 'random'):
            return lambda m, n : pd.DataFrame(m.random(n), columns=self.columns)
        else: 
            raise ValueError("No generate method found. Pass generate_fn explicitly.")
        
    def fit(self, data): 
        self._fit_fn(self.model, data)
        self._is_fitted=True 
        
    def generate(self, n_samples: int) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Model is not trained")
        try:
            result = self._generate_fn(self.model, n_samples)
        except Exception as e:
            raise RuntimeError(
                f"Generation failed for model '{self.method_name}'. "
                f"Consider passing generate_fn explicitly. Original error: {e}"
            )
        if self._postprocess_fn:
            result = self._postprocess_fn(result)
        return result
