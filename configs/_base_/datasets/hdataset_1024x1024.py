_base_ = './hdataset.py'

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True),

    dict(type='CreateAnnotations'),

    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True),

    dict(type='CreateAnnotations'),

    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_dataloader = None
test_dataloader = None

val_evaluator = None
test_evaluator = None
