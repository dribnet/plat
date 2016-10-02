from arghandler import ArgumentHandler
from plat.bin.plat_sample import plat_sample

handler = ArgumentHandler()
handler.set_subcommands( {'sample':plat_sample} )
handler.run()
# handler.run(['echo','hello','world']) # echo will be called and 'hello world' will be printed
