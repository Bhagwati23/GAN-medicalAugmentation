"""ganaug: GAN-based data augmentation for small medical image datasets (Team 33, project P33).

Module order = data flow = dependency order:
config -> data -> cnn, dcgan -> train_gan -> augment -> classifier -> fid -> experiments -> report (-> demo app)
"""
