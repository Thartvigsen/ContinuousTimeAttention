from dataset import InHospitalMortality

config = {
    "model"   : {
        "hidden_dim"          : 50,
        "embed_dim"           : 20,
        "dropout_probability" : 0.2,
        "n_layers"            : 1,
        "rnn_type"            : "GRU",
        "ortho_init"          : False,
        "noisin"              : False,
    },
    "training" : {
        "batch_size"          : 1,
        "n_epochs"            : 400,
        "learning_rate"       : 1e-4,
        "num_workers"         : 6,
        "use_scheduler"       : False,
        "scheduler_param"     : 0.95,
        "use_cuda"            : False,
        "resume"              : False,
        "loss_name"           : "crossentropy",
        "multilabel"          : True,
        "optimizer_name"      : "adam",
        "device"              : "cpu",
        "checkpoint"          : 50,
    },
}

d = InHospitalMortality()
