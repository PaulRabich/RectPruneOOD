from typing import List
from torch.nn import Module

from rect_prune_ood.lib.methods.layer_scaler import LayerScaler
from rect_prune_ood.lib.methods.rect_prune_methods import AshBPruningModule, AshPPruningModule, AshSPruningModule, DicePruningModule, ReactRectModule, BatsRectModule

module_dict = {
    "ashp": AshPPruningModule,
    "ashb": AshBPruningModule,
    "ashs": AshSPruningModule,
    "dice": DicePruningModule,
    "react": ReactRectModule,
    "bats": BatsRectModule,
    "layer-scaler": LayerScaler
}
def get_module_list(method_name:str) -> List[Module]:
    if method_name == "baseline":
        return []
    
    modules = []
    modules_str = method_name.split("_")

    for module_str in modules_str:
        module_params = module_str.split("@")
        if len(module_params) == 1:
            module = module_params[0]
            modules.append(module_dict[module]())
        elif len(module_params) == 2:
            module = module_params[0]
            param = float(module_params[1])
            modules.append(module_dict[module](param))
        else:
            raise ValueError(f"Invalid module string: {module_str}")
        
    return modules