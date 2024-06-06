
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'OTMorph':
        from .OTMorph_model import OTMorph
        model = OTMorph()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
