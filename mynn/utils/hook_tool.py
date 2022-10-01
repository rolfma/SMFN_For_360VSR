class HookTool:
    def __init__(self) -> None:
        self.hook_feat_out = dict()
        self.hook_feat_in = dict()
        self.name_module_map = dict()
        self.layer_name = None
        self.layer_names = set()

    def _get_hook_feat(self, module, feat_in, feat_out):
        self.hook_feat_out[module] = feat_out
        self.hook_feat_in[module] = feat_in

    def register_hook(self, model, layer_name, func='get_hook_feat'):
        """Register a hook for a specific layer.

        Args:
            model (nn.Module): The model.
            layer_name (str): Name of a specific layer.
            func (str): Set function for hook.
        """
        assert func in ['get_hook_feat'], 'Unknown function name.'
        func_dict = {'get_hook_feat': self._get_hook_feat}
        is_found = False
        for name, module in model.named_modules():
            if name == layer_name:
                self.name_module_map[layer_name] = module
                self.layer_name = layer_name
                self.layer_names.add(layer_name)
                is_found = True
                module.register_forward_hook(hook=func_dict['get_hook_feat'])
                break
        assert is_found == True, 'Can not found target layer.'

    def get_hook_feat(self, layer_name):
        """Get feature of a specific layer by hook.

        Args:
            layer_name (str): The name of a layer.

        Returns:
            tuple: output feature and input feature.
        """
        assert layer_name in self.layer_names, f'{layer_name} has no hook.'
        return self.hook_feat_out[self.name_module_map[
            layer_name]], self.hook_feat_in[self.name_module_map[layer_name]]
