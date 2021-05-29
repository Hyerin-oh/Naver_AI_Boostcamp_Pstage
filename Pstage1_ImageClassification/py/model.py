import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers.classifier import ClassifierHead
import torch.nn.utils.weight_norm as weightNorm

###############################################################
#                            BASE                             
###############################################################   

class EBaseModel(nn.Module):
    """
    EBaseModel : Efficientnet-b0 + No FC
    """
    def __init__(self , num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3' , pretrained = True)
        self.model._fc = nn.Linear(1536,num_classes)
        
    def forward(self,x):
        x = self.model(x)
        return x

    
class RBaseModel(nn.Module):
    """
    RBaseModel : Resnet50 + No FC
    """
    def __init__(self , num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained = 'True')
        self.ReLU = nn.ReLU(True)
        self.fc1 = nn.Linear(1000, num_classes)
        
    def forward(self,x):
        x = self.model(x)
        x = self.ReLU(x)
        x = self.fc1(x)
        return x
    
    
###############################################################
#                   specific task Model                       #
###############################################################    

class AgeModel(nn.Module):
    def __init__(self , num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.AgeHead = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features = 1000, out_features = num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.AgeHead(x)
        return x
    
class GenderModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.GenderHead = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.GenderHead(x)
        return x
    
class MaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.MaskHead = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1000, out_features = num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.MaskHead(x)
        return x
    
    
###############################################################
#                      MultisampleModel                       #
############################################################### 

class MultisampleModel(nn.Module):
    def __init__(self , num_classes, dropout_num = 4 ,dropout_p = 0.5):
        super().__init__()
        self.dropout_num = dropout_num
        self.backbone =  timm.create_model('efficientnet_b0' , pretrained = True)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.fc = nn.Linear(1000 , num_classes)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        x = self.backbone(x)
        for i,dropout in enumerate(self.dropouts):
            if i== 0:
                out = self.fc(dropout(x))
            else : 
                out += self.fc(dropout(x))

        return out / self.dropout_num


class MB0SimpleModel(nn.Module):
    
    """
    MultiBranchModel 
        - output (tuppe):([batch_size , 3] ,[batch_size , 2] , [batch_size , 3] )
    """
    
    def __init__(self, num_classes):
        super().__init__()
        model = timm.create_model('efficientnet_b0' , pretrained = True)
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-3]))
        n_feautres = 1280
        self.mask_classifier = ClassifierHead(n_feautres , 3)
        self.gender_classifier = ClassifierHead(n_feautres , 2)
        self.age_classifier = ClassifierHead(n_feautres , 3)

    def forward(self, x):
        x = self.backbone(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x) 
        return (mask, gender, age) 
    

class MultiBranchModel(nn.Module):
    """
    MultiBranchModel
        - output (tuppe):([batch_size , 3] ,[batch_size , 2] , [batch_size , 3] )
    """
    def __init__(self, num_classes, dropout_num = 8 ,dropout_p = 0.7):
        super().__init__()
        model = timm.create_model('efficientnet_b3' , pretrained = True)
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.dropout_num = dropout_num
        self.maskdropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.maskfc = nn.Linear(1536 , num_classes)
        nn.init.normal_(self.maskfc.weight, std=0.001)
        nn.init.constant_(self.maskfc.bias, 0)
        self.genderdropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.genderfc = nn.Linear(1536 , num_classes-1)
        nn.init.normal_(self.genderfc.weight, std=0.001)
        nn.init.constant_(self.genderfc.bias, 0)
        self.agedropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.agefc = nn.Linear(1536 , num_classes)
        nn.init.normal_(self.agefc.weight, std=0.001)
        nn.init.constant_(self.agefc.bias, 0)
        
    def forward(self, x):
        x = self.backbone(x)
        # Mask
        for i,dropout in enumerate(self.maskdropouts):
            if i== 0:
                mask = self.maskfc(dropout(x))
            else :
                mask += self.maskfc(dropout(x))
        mask /= self.dropout_num
        # Gender
        for i,dropout in enumerate(self.genderdropouts):
            if i== 0:
                gender = self.genderfc(dropout(x))
            else :
                gender += self.genderfc(dropout(x))
        gender /= self.dropout_num
        # Age
        for i,dropout in enumerate(self.agedropouts):
            if i== 0:
                age = self.agefc(dropout(x))
            else :
                age += self.agefc(dropout(x))
        age /= self.dropout_num
        return (mask, gender, age)
    
    
class Deit(nn.Module):
    """
    MultiBranchModel
        - output (tuppe):([batch_size , 3] ,[batch_size , 2] , [batch_size , 3] )
    """
    def __init__(self, num_classes, dropout_num = 8 ,dropout_p = 0.7):
        super().__init__()
        self.backbone = timm.create_model('vit_deit_small_patch16_224' , pretrained = True)
        self.dropout_num = dropout_num
        self.maskdropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.maskfc = nn.Linear(1000 , num_classes)
        nn.init.normal_(self.maskfc.weight, std=0.001)
        nn.init.constant_(self.maskfc.bias, 0)
        self.genderdropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.genderfc = nn.Linear(1000 , num_classes-1)
        nn.init.normal_(self.genderfc.weight, std=0.001)
        nn.init.constant_(self.genderfc.bias, 0)
        self.agedropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(dropout_num)])
        self.agefc = nn.Linear(1000 , num_classes)
        nn.init.normal_(self.agefc.weight, std=0.001)
        nn.init.constant_(self.agefc.bias, 0)
        
    def forward(self, x):
        x = self.backbone(x)
        # Mask
        for i,dropout in enumerate(self.maskdropouts):
            if i== 0:
                mask = self.maskfc(dropout(x))
            else :
                mask += self.maskfc(dropout(x))
        mask /= self.dropout_num
        # Gender
        for i,dropout in enumerate(self.genderdropouts):
            if i== 0:
                gender = self.genderfc(dropout(x))
            else :
                gender += self.genderfc(dropout(x))
        gender /= self.dropout_num
        # Age
        for i,dropout in enumerate(self.agedropouts):
            if i== 0:
                age = self.agefc(dropout(x))
            else :
                age += self.agefc(dropout(x))
        age /= self.dropout_num
        return (mask, gender, age)
        return x