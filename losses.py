
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# VisDrone has 10 classes
NUM_CLASSES = 10

def ib_loss(input_values, ib):
    """Computes the IB loss"""
    assert input_values.size(0) == ib.size(0), f"Batch size mismatch: input_values ({input_values.size(0)}) vs ib ({ib.size(0)})"
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        assert input.size(0) == target.size(0) == features.size(0), \
            f"Batch size mismatch: input ({input.size(0)}), target ({target.size(0)}), features ({features.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"
        assert features.dim() == 1, f"Expected 1D features, got shape {features.shape}"

        probs = F.softmax(input, dim=1)
        y_true = F.one_hot(target, NUM_CLASSES).float().to(input.device)
        grads = torch.sum(torch.abs(probs - y_true), dim=1)
        ib = grads * features
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    """Computes the IB Focal loss"""
    assert input_values.size(0) == ib.size(0), f"Batch size mismatch: input_values ({input_values.size(0)}) vs ib ({ib.size(0)})"
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        assert input.size(0) == target.size(0) == features.size(0), \
            f"Batch size mismatch: input ({input.size(0)}), target ({target.size(0)}), features ({features.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"
        assert features.dim() == 1, f"Expected 1D features, got shape {features.shape}"

        probs = F.softmax(input, dim=1)
        y_true = F.one_hot(target, NUM_CLASSES).float().to(input.device)
        grads = torch.sum(torch.abs(probs - y_true), dim=1)
        ib = grads * features
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

def focal_loss(input_values, gamma):
    """Computes the Focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.alpha=alpha

    def forward(self, input, target):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"
        return self.alpha*(focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma))

class AsymmetricFocalLoss_orig(nn.Module):
    def __init__(self, cls_num_list, alpha=0.25, gamma1=3., gamma2=2., weight=None, epsilon=1e-07):
        super(AsymmetricFocalLoss_orig, self).__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        # ce = -torch.log(p_true)
        ce=F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        # loss = torch.where(is_rare, self.delta * ce, (1 - self.delta) * torch.pow(1 - p_true, self.gamma) * ce)
        loss = torch.where(is_rare, torch.pow(1 - p, self.gamma1)* ce,  torch.pow(p, self.gamma2)*ce )
        return torch.mean(loss)
    
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, cls_num_list, alpha=0.25, gamma1=3., gamma2=2., weight=None, epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        # ce = -torch.log(p_true)
        ce=F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        # loss = torch.where(is_rare, self.delta * ce, (1 - self.delta) * torch.pow(1 - p_true, self.gamma) * ce)
        loss = torch.where(is_rare, self.alpha* torch.pow(1 - p, self.gamma1)* ce,  (1-self.alpha) * torch.pow(p, self.gamma2)*ce )
        return torch.mean(loss)

class SymmetricFocalTverskyLoss_orig(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75, weight=None, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss_orig, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        y_true = F.one_hot(target, NUM_CLASSES).float().to(input.device)

        tversky_loss = []
        for c in range(NUM_CLASSES):
            tp = torch.sum(y_true[:, c] * y_pred[:, c], dim=0)
            fn = torch.sum(y_true[:, c] * (1 - y_pred[:, c]), dim=0)
            fp = torch.sum((1 - y_true[:, c]) * y_pred[:, c], dim=0)
            tversky = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
            tversky_loss.append(torch.pow(1 - tversky, self.gamma))

        loss = torch.mean(torch.stack(tversky_loss))
        return loss

class SymmetricFocalTverskyLoss_mod(nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, weight=None, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss_mod, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        y_true = F.one_hot(target, NUM_CLASSES).float().to(input.device)

        tversky_loss = []
        for c in range(NUM_CLASSES):
            tp = torch.sum(y_true[:, c] * y_pred[:, c], dim=0)
            fn = torch.sum(y_true[:, c] * (1 - y_pred[:, c]), dim=0)
            fp = torch.sum((1 - y_true[:, c]) * y_pred[:, c], dim=0)
            tversky = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
            tversky_loss.append((1 - tversky) * torch.pow(1 - tversky, -self.gamma))

        loss = torch.mean(torch.stack(tversky_loss))
        return loss
    


class AsymmetricFocalTverskyLoss_mod(nn.Module):
    def __init__(self, cls_num_list, delta=0.6, gamma=0.5, weight=None, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss_mod, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        y_true = F.one_hot(target, NUM_CLASSES).float().to(input.device)

        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes

        tversky_loss = []
        for c in range(NUM_CLASSES):
            tp = torch.sum(y_true[:, c] * y_pred[:, c], dim=0)
            fn = torch.sum(y_true[:, c] * (1 - y_pred[:, c]), dim=0)
            fp = torch.sum((1 - y_true[:, c]) * y_pred[:, c], dim=0)
            tversky = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
            if c in effective_rare_classes:
                tversky_loss.append((1 - tversky) * torch.pow(1 - tversky, 1 - self.gamma))
            else:
                tversky_loss.append(1 - tversky)

        loss = torch.mean(torch.stack(tversky_loss))
        return loss

class SymmetricFocalLoss_mod(nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, weight=None, epsilon=1e-07):
        super(SymmetricFocalLoss_mod, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        ce = -torch.log(p_true)
        focal_term = torch.pow(1 - p_true, (1-self.gamma))
        # delta_factor = torch.where(target == 0, 1 - self.delta, self.delta)
        loss = ce * focal_term * self.delta
        return torch.mean(loss)
    
class AsymmetricFocalLoss_mod(nn.Module):
    def __init__(self, cls_num_list, gamma1=0.5,delta1=0.6, weight=None, epsilon=1e-07):
        super(AsymmetricFocalLoss_mod, self).__init__()
        self.gamma1 = gamma1
        # self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        self.delta1=delta1
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"
        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        ce = -torch.log(p_true)
        # p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        loss = torch.where(is_rare, self.delta1 * ce, (1 - self.delta1) * torch.pow(1 - p_true, self.gamma1) * ce)
        # loss = torch.where(is_rare, self.alpha* torch.pow(1 - p_true, self.gamma1)* torch.log(p_true),  (1-self.alpha) * torch.pow(p_true, self.gamma2)*torch.log(1-p_true) )
        return torch.mean(loss)
    
class SymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.focal_tversky = SymmetricFocalTverskyLoss_mod(delta=delta, gamma=gamma)
        self.focal = SymmetricFocalLoss_mod(delta=delta, gamma=gamma)

    def forward(self, input, target):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        symmetric_ftl = self.focal_tversky(input, target)
        symmetric_fl = self.focal(input, target)
        return (self.weight * symmetric_fl) + ((1 - self.weight) * symmetric_ftl)

class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, cls_num_list, weight=0.5, delta=0.6, gamma=0.5):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]
        self.focal_tversky = AsymmetricFocalTverskyLoss_mod(cls_num_list, delta=delta, gamma=gamma)
        self.focal = AsymmetricFocalLoss_mod(cls_num_list, gamma1=gamma, delta1=delta)

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes

        asymmetric_ftl = self.focal_tversky(input, target, rare_classes=effective_rare_classes)
        asymmetric_fl = self.focal(input, target, rare_classes=effective_rare_classes)
        return (self.weight * asymmetric_fl) + ((1 - self.weight) * asymmetric_ftl)




class AUFL_SUFL(nn.Module):
    def __init__(self, cls_num_list, weight=0.5, delta=0.6, gamma=0.5, rare_threshold=100):
        super(AUFL_SUFL, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"

        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.symmetric_loss = SymmetricUnifiedFocalLoss(weight=weight, delta=delta, gamma=gamma)
        self.asymmetric_loss = AsymmetricUnifiedFocalLoss(cls_num_list, weight=weight, delta=delta, gamma=gamma)

    def forward(self, input, target):
        """
        Chooses asymmetric loss if target contains rare classes, 
        otherwise chooses symmetric loss.
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Find which classes are present in target
        target_classes = torch.unique(torch.argmax(target)).cpu().numpy().tolist()

        # Check if any rare class exists in target
        if any(cls in self.rare_classes for cls in target_classes):
            return self.asymmetric_loss(input, target, rare_classes=self.rare_classes)
        else:
            return self.symmetric_loss(input, target)



class AUFL_FL(nn.Module):
    def __init__(self, cls_num_list, weight=0.5, delta=0.6, gamma=0.5, rare_threshold=100, alpha=0.25):
        super(AUFL_FL, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.alpha = alpha
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"

        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.asymmetric_loss = AsymmetricUnifiedFocalLoss(cls_num_list, weight=weight, delta=delta, gamma=gamma)
        self.focal_loss = FocalLoss(weight=None, alpha=alpha)

    def forward(self, input, target):
        """
        Chooses asymmetric unified focal loss if target contains rare classes,
        otherwise uses standard focal loss.
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Extract classes from target
        if target.dim() > 1 and target.size(1) == NUM_CLASSES:  
            # one-hot encoded
            target_classes = torch.unique(torch.argmax(target, dim=1)).cpu().numpy().tolist()
        else:  
            # class indices
            target_classes = torch.unique(target).cpu().numpy().tolist()

        # Check if batch contains rare classes
        if any(cls in self.rare_classes for cls in target_classes):
            return self.asymmetric_loss(input, target, rare_classes=self.rare_classes)
        else:
            return self.focal_loss(input, target)




class AFL_FL(nn.Module):
    def __init__(self, cls_num_list, rare_threshold=100, alpha=0.25, gamma=2., gamma1=3., gamma2=2., weight=None, epsilon=1e-7):
        super(AFL_FL, self).__init__()
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        
        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma, alpha=alpha)
        self.asym_focal_loss = AsymmetricFocalLoss(cls_num_list, alpha=alpha, gamma1=gamma1, gamma2=gamma2, weight=weight, epsilon=epsilon)

    def forward(self, input, target):
        """
        If batch contains rare classes -> use AsymmetricFocalLoss
        Else -> use standard FocalLoss
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Extract classes from target
        if target.dim() > 1 and target.size(1) == NUM_CLASSES:  
            # one-hot encoded
            target_classes = torch.unique(torch.argmax(target, dim=1)).cpu().numpy().tolist()
        else:  
            # class indices
            target_classes = torch.unique(target).cpu().numpy().tolist()

        # Decide which loss to use
        if any(cls in self.rare_classes for cls in target_classes):
            return self.asym_focal_loss(input, target, rare_classes=self.rare_classes)
        else:
            return self.focal_loss(input, target)





class AFL_SUFL(nn.Module):
    def __init__(self, cls_num_list, rare_threshold=100, weight=0.5, delta=0.6, gamma=0.5,
                 alpha=0.25, gamma1=3., gamma2=2., epsilon=1e-7):
        super(AFL_SUFL, self).__init__()
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"

        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.asym_focal_loss = AsymmetricFocalLoss(
            cls_num_list=cls_num_list,
            alpha=alpha, gamma1=gamma1, gamma2=gamma2,
            weight=None, epsilon=epsilon
        )
        self.sym_ufo_loss = SymmetricUnifiedFocalLoss(weight=weight, delta=delta, gamma=gamma)

    def forward(self, input, target):
        """
        If batch contains rare classes -> use AsymmetricFocalLoss
        Else -> use SymmetricUnifiedFocalLoss
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Extract classes from target
        if target.dim() > 1 and target.size(1) == NUM_CLASSES:  
            # one-hot encoded
            target_classes = torch.unique(torch.argmax(target, dim=1)).cpu().numpy().tolist()
        else:  
            # class indices
            target_classes = torch.unique(target).cpu().numpy().tolist()

        # Decide which loss to use
        if any(cls in self.rare_classes for cls in target_classes):
            return self.asym_focal_loss(input, target, rare_classes=self.rare_classes)
        else:
            return self.sym_ufo_loss(input, target)




class AFL_FL_mod(nn.Module):
    def __init__(self, cls_num_list, rare_threshold=100, alpha=0.25, gamma=2., gamma1=3., gamma2=2., weight=None, epsilon=1e-7):
        super(AFL_FL_mod, self).__init__()
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        
        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma, alpha=alpha)
        self.asym_focal_loss = AsymmetricFocalLoss(cls_num_list, alpha=alpha, gamma1=gamma1, gamma2=gamma2, weight=weight, epsilon=epsilon)

    def forward(self, input, target):
        """
        If batch contains rare classes -> use AsymmetricFocalLoss
        Else -> use standard FocalLoss
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Extract classes from target
        if target.dim() > 1 and target.size(1) == NUM_CLASSES:  
            # one-hot encoded
            target_classes = torch.unique(torch.argmax(target, dim=1)).cpu().numpy().tolist()
        else:  
            # class indices
            target_classes = torch.unique(target).cpu().numpy().tolist()

        # Decide which loss to use
        if any(cls in self.rare_classes for cls in target_classes):
            return self.focal_loss(input, target)
        else:
            return self.asym_focal_loss(input, target, rare_classes=self.rare_classes)


class AFL_SUFL_mod(nn.Module):
    def __init__(self, cls_num_list, rare_threshold=100, weight=0.5, delta=0.6, gamma=0.5,
                 alpha=0.25, gamma1=3., gamma2=2., epsilon=1e-7):
        super(AFL_SUFL_mod, self).__init__()
        self.rare_threshold = rare_threshold

        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"

        # Identify rare classes
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < rare_threshold]

        # Sub-losses
        self.asym_focal_loss = AsymmetricFocalLoss(
            cls_num_list=cls_num_list,
            alpha=alpha, gamma1=gamma1, gamma2=gamma2,
            weight=None, epsilon=epsilon
        )
        self.sym_ufo_loss = SymmetricUnifiedFocalLoss(weight=weight, delta=delta, gamma=gamma)

    def forward(self, input, target):
        """
        If batch contains rare classes -> use AsymmetricFocalLoss
        Else -> use SymmetricUnifiedFocalLoss
        """
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        # Extract classes from target
        if target.dim() > 1 and target.size(1) == NUM_CLASSES:  
            # one-hot encoded
            target_classes = torch.unique(torch.argmax(target, dim=1)).cpu().numpy().tolist()
        else:  
            # class indices
            target_classes = torch.unique(target).cpu().numpy().tolist()

        # Decide which loss to use
        if any(cls in self.rare_classes for cls in target_classes):
            return self.sym_ufo_loss(input, target)
        else:
            return self.asym_focal_loss(input, target, rare_classes=self.rare_classes)

class AFL_mod1(nn.Module):
    def __init__(self, cls_num_list, alpha=0.25, gamma1=3., gamma2=2., weight=None, epsilon=1e-07):
        super(AFL_mod1, self).__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        # ce = -torch.log(p_true)
        ce=F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        # loss = torch.where(is_rare, self.delta * ce, (1 - self.delta) * torch.pow(1 - p_true, self.gamma) * ce)
        loss = torch.where(is_rare, self.alpha* torch.pow(1 - p, self.gamma1)* ce,  (1-self.alpha) * torch.pow(1-p, self.gamma2)*ce )
        return torch.mean(loss)
    
class AFL_mod2(nn.Module):
    def __init__(self, cls_num_list, alpha=0.25, gamma1=3., gamma2=2., weight=None, epsilon=1e-07):
        super(AFL_mod2, self).__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        # ce = -torch.log(p_true)
        ce=F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        # loss = torch.where(is_rare, self.delta * ce, (1 - self.delta) * torch.pow(1 - p_true, self.gamma) * ce)
        loss = torch.where(is_rare, (1-self.alpha)* torch.pow(1 - p, self.gamma1)* ce,  self.alpha * torch.pow(1-p, self.gamma2)*ce )
        return torch.mean(loss)

class AFL_mod3(nn.Module):
    def __init__(self, cls_num_list, alpha=0.25, gamma1=3., gamma2=2., weight=None, epsilon=1e-07):
        super(AFL_mod3, self).__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2=gamma2
        self.weight = weight
        self.epsilon = epsilon
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        assert len(cls_num_list) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(cls_num_list)}"
        self.rare_classes = [i for i, count in enumerate(cls_num_list) if count < 100]

    def forward(self, input, target, rare_classes=None):
        assert input.size(0) == target.size(0), f"Batch size mismatch: input ({input.size(0)}) vs target ({target.size(0)})"
        assert input.size(1) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {input.size(1)}"

        batch_size = input.size(0)
        y_pred = F.softmax(input, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        p_true = y_pred[range(batch_size), target]
        # ce = -torch.log(p_true)
        ce=F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p=torch.exp(-ce)
        effective_rare_classes = rare_classes if rare_classes is not None else self.rare_classes
        is_rare = torch.tensor([t.item() in effective_rare_classes for t in target], dtype=torch.bool).to(input.device)
        # loss = torch.where(is_rare, self.delta * ce, (1 - self.delta) * torch.pow(1 - p_true, self.gamma) * ce)
        loss = torch.where(is_rare, (self.alpha)* torch.pow(1 - p, self.gamma1)* ce,  (1-self.alpha) * (1-p)*ce )
        return torch.mean(loss)


if __name__ == "__main__":
    batch_size = 16
    input = torch.randn(batch_size, NUM_CLASSES)
    target = torch.randint(0, NUM_CLASSES, (batch_size,))
    features = torch.ones(batch_size)
    cls_num_list = [1375, 188, 21, 4363, 82, 94, 18, 3, 32, 295]

    # Test IBLoss
    ib_loss_fn = IBLoss()
    loss = ib_loss_fn(input, target, features)
    print(f"IBLoss: {loss.item()}")

    # Test IB_FocalLoss
    ib_focal_loss_fn = IB_FocalLoss()
    loss = ib_focal_loss_fn(input, target, features)
    print(f"IB_FocalLoss: {loss.item()}")

    # Test FocalLoss
    focal_loss_fn = FocalLoss()
    loss = focal_loss_fn(input, target)
    print(f"FocalLoss: {loss.item()}")

   
# Test FocalLoss
    asym_orig_loss_fn = AsymmetricFocalLoss_orig()
    loss = asym_orig_loss_fn(input, target)
    print(f"AsymmetricFocalLoss_orig: {loss.item()}")

# Test FocalLoss
    asym_loss_fn = AsymmetricFocalLoss()
    loss = asym_loss_fn(input, target)
    print(f"AsymmetricFocalLoss: {loss.item()}")

    # Test SymmetricFocalTverskyLoss
    sym_tversky_loss_fn = SymmetricFocalTverskyLoss_orig()
    loss = sym_tversky_loss_fn(input, target)
    print(f"SymmetricFocalTverskyLoss_orig: {loss.item()}")


    # Test SymmetricFocalTverskyLoss
    sym_tversky_loss_fn = SymmetricFocalTverskyLoss_mod()
    loss = sym_tversky_loss_fn(input, target)
    print(f"SymmetricFocalTverskyLoss_mod: {loss.item()}")

    # Test AsymmetricFocalTverskyLoss
    asym_tversky_loss_fn = AsymmetricFocalTverskyLoss_mod(cls_num_list=cls_num_list)
    loss = asym_tversky_loss_fn(input, target)
    print(f"AsymmetricFocalTverskyLoss_mod: {loss.item()}")

    # Test SymmetricUnifiedFocalLoss
    sym_unified_loss_fn = SymmetricUnifiedFocalLoss()
    loss = sym_unified_loss_fn(input, target)
    print(f"SymmetricUnifiedFocalLoss: {loss.item()}")

    # Test AsymmetricUnifiedFocalLoss
    asym_unified_loss_fn = AsymmetricUnifiedFocalLoss(cls_num_list=cls_num_list)
    loss = asym_unified_loss_fn(input, target)
    print(f"AsymmetricUnifiedFocalLoss: {loss.item()}")

        # Test AUFL_SUFL
    aufl_sufl_loss_fn = AUFL_SUFL(cls_num_list=cls_num_list)
    loss = aufl_sufl_loss_fn(input, target)
    print(f"AUFL_SUFL: {loss.item()}")

    # Test AUFL_FL
    aufl_fl_loss_fn = AUFL_FL(cls_num_list=cls_num_list)
    loss = aufl_fl_loss_fn(input, target)
    print(f"AUFL_FL: {loss.item()}")

    # Test AFL_FL
    afl_fl_loss_fn = AFL_FL(cls_num_list=cls_num_list)
    loss = afl_fl_loss_fn(input, target)
    print(f"AFL_FL: {loss.item()}")

    # Test AFL_SUFL
    afl_sufl_loss_fn = AFL_SUFL(cls_num_list=cls_num_list)
    loss = afl_sufl_loss_fn(input, target)
    print(f"AFL_SUFL: {loss.item()}")

    # Test AFL_FL_mod
    afl_fl_mod_loss_fn = AFL_FL_mod(cls_num_list=cls_num_list)
    loss = afl_fl_mod_loss_fn(input, target)
    print(f"AFL_FL_mod: {loss.item()}")

     # Test AFL_SUFL_mod
    afl_sufl_mod_loss_fn = AFL_SUFL_mod(cls_num_list=cls_num_list)
    loss = afl_sufl_mod_loss_fn(input, target)
    print(f"AFL_SUFL_mod: {loss.item()}")

         # Test AFL_mod1
    afl_mod1_loss_fn = AFL_mod1(cls_num_list=cls_num_list)
    loss = afl_mod1_loss_fn(input, target)
    print(f"AFL_mod1: {loss.item()}")

     # Test AFL_mod2
    afl_mod2_loss_fn = AFL_mod2(cls_num_list=cls_num_list)
    loss = afl_mod2_loss_fn(input, target)
    print(f"AFL_mod2: {loss.item()}")

     # Test AFL_mod3
    afl_mod3_loss_fn = AFL_mod3(cls_num_list=cls_num_list)
    loss = afl_mod3_loss_fn(input, target)
    print(f"AFL_mod3: {loss.item()}")