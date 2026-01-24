################
## CHANGE NAME OF ANSI COMPILER HERE
################
# Using gcc as the standard ANSI compiler for Linux
CC      = gcc

# Current values for DATABASE are: INFORMIX, DB2, TDAT (Teradata)
#                                  SQLSERVER, SYBASE, ORACLE, VECTORWISE
# Current values for MACHINE are:  ATT, DOS, HP, IBM, ICL, MVS,
#                                  SGI, SUN, U2200, VMS, LINUX, WIN32
# Current values for WORKLOAD are: TPCH

DATABASE = VECTORWISE
MACHINE  = LINUX
WORKLOAD = TPCH

# --- Configuration Flags ---
# Includes necessary defines for the database, machine, and workload
CFLAGS   = -g -DDBNAME=\"dss\" -D$(MACHINE) -D$(DATABASE) -D$(WORKLOAD) -DRNG_TEST -D_FILE_OFFSET_BITS=64
LDFLAGS  = -O
OBJ      = .o
EXE      =
LIBS     = -lm

###############
## NO CHANGES SHOULD BE NECESSARY BELOW THIS LINE
###############

VERSION=2
RELEASE=13
PATCH=0

# Program and Source Definitions
PROG1 = dbgen$(EXE)
PROG2 = qgen$(EXE)
PROGS = $(PROG1) $(PROG2)

HDR1 = dss.h rnd.h config.h dsstypes.h shared.h bcd2.h rng64.h release.h
HDR2 = tpcd.h permute.h
HDR  = $(HDR1) $(HDR2)

SRC1 = build.c driver.c bm_utils.c rnd.c print.c load_stub.c bcd2.c speed_seed.c text.c permute.c rng64.c
SRC2 = qgen.c varsub.c

OBJ1 = build$(OBJ) driver$(OBJ) bm_utils$(OBJ) rnd$(OBJ) print$(OBJ) load_stub$(OBJ) bcd2$(OBJ) speed_seed$(OBJ) text$(OBJ) permute$(OBJ) rng64$(OBJ)
OBJ2 = build$(OBJ) bm_utils$(OBJ) qgen$(OBJ) rnd$(OBJ) varsub$(OBJ) text$(OBJ) bcd2$(OBJ) permute$(OBJ) speed_seed$(OBJ) rng64$(OBJ)

SETS = dists.dss

# --- Build Targets ---
all: $(PROGS)

$(PROG1): $(OBJ1) $(SETS)
        $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ1) $(LIBS)

$(PROG2): permute.h $(OBJ2)
        $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ2) $(LIBS)

clean:
        rm -f $(PROGS) $(OBJS)

rnd$(OBJ): rnd.h
$(OBJ1): $(HDR1)
$(OBJ2): dss.h tpcd.h config.h rng64.h release.h

