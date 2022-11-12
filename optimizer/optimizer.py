from transformers import AdamW, get_linear_schedule_with_warmup


def get_scheduler_and_optimizer_(args, model, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = model.bert.named_parameters()
    classifier_addressEntity_parameters = model.addressEntity.named_parameters()
    classifier_influence_parameters = model.influence.named_parameters()
    classifier_requestion_parameters = model.requestion.named_parameters()
    classifier_houseType_parameters = model.houseType_cls.named_parameters()
    classifier_num_parameters = model.num_cls.named_parameters()

    crf_houseType_parameters = model.houseType_crf.named_parameters()
    crf_num_parameters = model.num_crf.named_parameters()

    args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
    args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
    args.crf_lr = args.crf_lr if args.crf_lr else args.learning_rate

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.bert_lr},

        {"params": [p for n, p in classifier_addressEntity_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_addressEntity_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in classifier_influence_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_influence_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in classifier_requestion_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_requestion_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in classifier_houseType_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_houseType_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in classifier_num_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.classifier_lr},
        {"params": [p for n, p in classifier_num_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.classifier_lr},

        {"params": [p for n, p in crf_houseType_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.crf_lr},
        {"params": [p for n, p in crf_houseType_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.crf_lr},

        {"params": [p for n, p in crf_num_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         "lr": args.crf_lr},
        {"params": [p for n, p in crf_num_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.crf_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def get_scheduler_and_optimizer(args, model, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = model.bert.named_parameters()
    linears_parameters = model.modelist.named_parameters()
    crfs_parameters = model.crf_layer.named_parameters()

    args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
    args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
    args.crf_lr = args.crf_lr if args.crf_lr else args.learning_rate

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.bert_lr},

        {"params": [p for n, p in linears_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.classifier_lr},
        {"params": [p for n, p in linears_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.classifier_lr},

        {"params": [p for n, p in crfs_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": args.crf_lr},
        {"params": [p for n, p in crfs_parameters if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.crf_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler
