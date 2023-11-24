from invoke import task

@task
def build(ctx, install=False):
    print("Building the extension")
    if install:
        ctx.run("flit install")
    else:
        ctx.run("flit build")
