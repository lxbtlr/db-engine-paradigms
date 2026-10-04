// Minimal QEMU TCG plugin: total guest instructions executed (single
// vCPU, user mode). Prints "insns <n>" to stderr at exit. Used by
// study/neon_port/insns.sh for dynamic instructions per element of each
// run_neonbench impl (the QEMU in flake.nix's qemu-user ships no plugins).
#include <qemu-plugin.h>
#include <stdio.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static struct qemu_plugin_scoreboard* sb;
static qemu_plugin_u64 count;

static void tb_trans(struct qemu_plugin_tb* tb, void* ud) {
   (void)ud;
   qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
       tb, QEMU_PLUGIN_INLINE_ADD_U64, count, qemu_plugin_tb_n_insns(tb));
}

static void at_exit(void* p) {
   (void)p;
   fprintf(stderr, "insns %llu\n", (unsigned long long)qemu_plugin_u64_sum(count));
   qemu_plugin_scoreboard_free(sb);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t* info,
                                           int argc, char** argv) {
   (void)info, (void)argc, (void)argv;
   sb = qemu_plugin_scoreboard_new(sizeof(uint64_t));
   count = qemu_plugin_scoreboard_u64(sb);
   qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans, NULL);
   qemu_plugin_register_atexit_cb(id, at_exit, NULL);
   return 0;
}
