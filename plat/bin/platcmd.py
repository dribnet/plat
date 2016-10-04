from arghandler import ArgumentHandler
from plat.bin.sample import sample
from plat.bin.atvec import atvec
from plat.bin.download import download

def main(args=None):
    handler = ArgumentHandler()
    handler.set_subcommands(
        {'sample': sample,
         'download': download,
         'atvec': atvec} )
    handler.run()

if __name__ == "__main__":
    main()
