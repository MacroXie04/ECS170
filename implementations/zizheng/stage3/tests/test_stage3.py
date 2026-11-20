"""Check image layout, labels, gradients, selection and checkpoint restoration."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Configurations import configurations
from local_code.stage_3_code.Method_CNN import augment
from local_code.stage_3_code.Setting import Setting, MODELS


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_orl_layout_and_original_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            def sample(label):
                return dict(image=np.full((112,92,3),128,np.uint8),label=label)
            with (Path(folder)/'ORL').open('wb') as stream:
                pickle.dump(dict(train=[sample(1),sample(40)],test=[sample(40)]),stream)
            loader=Dataset_Loader(folder,'ORL')
            train=loader.load()
            test=loader.load('test',train['classes'])
            self.assertEqual(tuple(train['X'].shape),(2,1,112,92))
            self.assertEqual(train['y'].tolist(),[0,1])
            self.assertEqual(test['y'].tolist(),[1])
            self.assertAlmostEqual(train['X'][0,0,0,0].item(),128/255)

    def test_model_shapes_modes_and_gradients(self):
        for name,shape in [('MNIST',(2,1,28,28)),('ORL',(2,1,112,92)),('CIFAR',(2,3,32,32))]:
            for config in configurations(name).values():
                model=MODELS[name](config=config)
                logits=model(torch.randn(shape))
                self.assertEqual(tuple(logits.shape),(2,40 if name=='ORL' else 10))
                model.criterion()(logits,torch.tensor([0,1])).backward()
                self.assertGreater(model.features[0].weight.grad.abs().sum().item(),0)
                model.eval()
                self.assertFalse(model.training)
                model.train()
                self.assertTrue(model.training)

    def test_augmentation_keeps_samples_separate(self):
        x=torch.arange(8).float()[:,None,None,None].expand(8,3,32,32)
        out=augment(x)
        self.assertEqual(tuple(out.shape),tuple(x.shape))
        self.assertTrue(torch.equal(out,x))

    def test_selection_refit_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            samples=[dict(image=np.full((28,28),i%2*255,np.float32),label=i%2) for i in range(40)]
            with (root/'MNIST').open('wb') as stream:
                pickle.dump(dict(train=samples,test=samples[:8]),stream)
            runner=Setting(Dataset_Loader(root,'MNIST'),root/'results')
            cfg=dict(configurations('MNIST')['baseline'],channels=[2,4],grid=[2,2],hidden=8,epochs=2,batch_size=8)
            record=runner.load_run_save_evaluate('baseline',cfg)
            with self.assertRaises(FileExistsError):
                runner.load_run_save_evaluate('baseline',cfg)
            runner.select(['baseline'])
            final=runner.final()
            checkpoint=torch.load(root/'results/final/model.pt',weights_only=True)
            model=MODELS['MNIST'](num_classes=2,config=checkpoint['config'])
            model.load_state_dict(checkpoint['state_dict'])
            test=runner.dataset.load('test',np.array(checkpoint['classes']))
            with np.load(root/'results/final/predictions.npz') as saved:
                np.testing.assert_array_equal(model.test(test['X']),saved['test_pred'])
            self.assertEqual(final['test_rows'],8)
            self.assertEqual(final['training_rows'],40)
            self.assertEqual(record['validation_rows'],4)


if __name__=='__main__':
    unittest.main()
