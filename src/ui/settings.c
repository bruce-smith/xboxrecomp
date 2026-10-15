/*
 * settings.c -- options declared in code, stored in an ini file.
 *
 *   [section]
 *   key = value
 *
 * A value is a number, or for a choice the option's name. Entries keep the
 * order they were first seen in (file, then declarations), so the file stays
 * stable across saves. An environment variable named for an option overrides
 * the file: agents and scripts set variables, people use the menu.
 */
#include "xbox_ui.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <strings.h>
#define _stricmp strcasecmp
#endif

#if defined(_WIN32)
#include <windows.h>
static CRITICAL_SECTION s_lock;
static INIT_ONCE s_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK lock_init(PINIT_ONCE o, PVOID p, PVOID *c)
{
    (void)o; (void)p; (void)c;
    InitializeCriticalSection(&s_lock);
    return TRUE;
}
#define LOCK()   do { InitOnceExecuteOnce(&s_once, lock_init, NULL, NULL); \
                      EnterCriticalSection(&s_lock); } while (0)
#define UNLOCK() LeaveCriticalSection(&s_lock)
#else
#define LOCK()   ((void)0)
#define UNLOCK() ((void)0)
#endif

enum { T_RAW, T_BOOL, T_INT, T_CHOICE };

typedef struct {
    char section[32], key[64], value[96];
    const char *label, *env;
    const char *const *names;
    int type, n, lo, hi, def;
    unsigned flags;
} Setting;

#define MAX_SETTINGS 1024
static Setting s_set[MAX_SETTINGS];
static int s_count;
static char s_path[512];

static Setting *find(const char *section, const char *key)
{
    int i;
    for (i = 0; i < s_count; i++)
        if (!strcmp(s_set[i].section, section) && !strcmp(s_set[i].key, key))
            return &s_set[i];
    return NULL;
}

static Setting *add(const char *section, const char *key)
{
    Setting *s = find(section, key);
    if (s || s_count == MAX_SETTINGS)
        return s;
    s = &s_set[s_count++];
    memset(s, 0, sizeof *s);
    snprintf(s->section, sizeof s->section, "%s", section);
    snprintf(s->key, sizeof s->key, "%s", key);
    return s;
}

static char *trim(char *p)
{
    char *e;
    while (isspace((unsigned char)*p))
        p++;
    e = p + strlen(p);
    while (e > p && isspace((unsigned char)e[-1]))
        *--e = 0;
    return p;
}

/* A stored or env string as the option's value. */
static int parse(const Setting *s, const char *v, int *out)
{
    char *end;
    long n;
    int i;
    if (!v || !*v)
        return 0;
    if (s->type == T_CHOICE)
        for (i = 0; i < s->n; i++)
            if (!_stricmp(v, s->names[i])) {
                *out = i;
                return 1;
            }
    if (s->type == T_BOOL) {
        if (!_stricmp(v, "true") || !_stricmp(v, "on") || !_stricmp(v, "yes")) {
            *out = 1;
            return 1;
        }
        if (!_stricmp(v, "false") || !_stricmp(v, "off") || !_stricmp(v, "no")) {
            *out = 0;
            return 1;
        }
    }
    n = strtol(v, &end, 0);
    if (end == v)
        return 0;
    if (s->type == T_BOOL)
        n = n != 0;
    if (s->type == T_INT && s->hi > s->lo) {
        if (n < s->lo) n = s->lo;
        if (n > s->hi) n = s->hi;
    }
    if (s->type == T_CHOICE && (n < 0 || n >= s->n))
        return 0;
    *out = (int)n;
    return 1;
}

static int value_of(const Setting *s)
{
    int v = s->def;
    if (s->env) {
        const char *e = getenv(s->env);
        if (e && parse(s, e, &v))
            return v;
    }
    if (parse(s, s->value, &v))
        return v;
    return s->def;
}

static void store(Setting *s, int v)
{
    if (s->type == T_CHOICE && v >= 0 && v < s->n)
        snprintf(s->value, sizeof s->value, "%s", s->names[v]);
    else
        snprintf(s->value, sizeof s->value, "%d", v);
}

void xbox_settings_load(const char *path)
{
    FILE *f;
    char line[256], section[32] = "";
    LOCK();
    snprintf(s_path, sizeof s_path, "%s", path);
    f = fopen(path, "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            char *p = trim(line), *eq;
            if (!*p || *p == ';' || *p == '#')
                continue;
            if (*p == '[') {
                char *e = strchr(p, ']');
                if (e) {
                    *e = 0;
                    snprintf(section, sizeof section, "%s", trim(p + 1));
                }
                continue;
            }
            eq = strchr(p, '=');
            if (!eq || !section[0])
                continue;
            *eq = 0;
            {
                Setting *s = add(section, trim(p));
                if (s)
                    snprintf(s->value, sizeof s->value, "%s", trim(eq + 1));
            }
        }
        fclose(f);
        fprintf(stderr, "  [SETTINGS] %s: %d entries\n", path, s_count);
    } else {
        fprintf(stderr, "  [SETTINGS] %s not found; defaults\n", path);
    }
    UNLOCK();
}

void xbox_settings_save(void)
{
    FILE *f;
    int i, j;
    LOCK();
    if (!s_path[0] || !(f = fopen(s_path, "w"))) {
        UNLOCK();
        return;
    }
    fprintf(f, "; Settings, edited by the in-game menu (Escape). An environment\n"
               "; variable named for a setting overrides it.\n");
    for (i = 0; i < s_count; i++) {
        int first = 1;
        for (j = 0; j < i; j++)
            if (!strcmp(s_set[j].section, s_set[i].section)) {
                first = 0;
                break;
            }
        if (!first)
            continue;
        fprintf(f, "\n[%s]\n", s_set[i].section);
        for (j = i; j < s_count; j++)
            if (!strcmp(s_set[j].section, s_set[i].section) && s_set[j].value[0])
                fprintf(f, "%s = %s\n", s_set[j].key, s_set[j].value);
    }
    fclose(f);
    UNLOCK();
}

static int declare(const char *section, const char *key, const char *label,
                   int type, int def, int lo, int hi, const char *const *names,
                   int n, const char *env, unsigned flags)
{
    Setting *s;
    int v;
    LOCK();
    s = add(section, key);
    if (!s) {
        UNLOCK();
        return def;
    }
    s->label = label;
    s->type = type;
    s->def = def;
    s->lo = lo;
    s->hi = hi;
    s->names = names;
    s->n = n;
    s->env = env;
    s->flags = flags;
    v = value_of(s);
    UNLOCK();
    return v;
}

int xbox_setting_bool(const char *section, const char *key, const char *label,
                      int def, const char *env, unsigned flags)
{
    return declare(section, key, label, T_BOOL, def != 0, 0, 1, NULL, 0, env, flags);
}

int xbox_setting_int(const char *section, const char *key, const char *label,
                     int def, int lo, int hi, const char *env, unsigned flags)
{
    return declare(section, key, label, T_INT, def, lo, hi, NULL, 0, env, flags);
}

int xbox_setting_choice(const char *section, const char *key, const char *label,
                        const char *const *names, int n, int def,
                        const char *env, unsigned flags)
{
    return declare(section, key, label, T_CHOICE, def, 0, n - 1, names, n, env, flags);
}

int xbox_setting_get(const char *section, const char *key, int def)
{
    Setting *s;
    int v = def;
    LOCK();
    s = find(section, key);
    if (s)
        v = s->type == T_RAW ? (parse(s, s->value, &v) ? v : def) : value_of(s);
    UNLOCK();
    return v;
}

void xbox_setting_set(const char *section, const char *key, int value)
{
    Setting *s;
    LOCK();
    s = add(section, key);
    if (s)
        store(s, value);
    UNLOCK();
}

int xbox_setting_to_env(const char *section, const char *key)
{
    Setting *s;
    char buf[32];
    int v, done = 0;
    LOCK();
    s = find(section, key);
    if (s && s->env && !getenv(s->env) && s->value[0]) {
        v = value_of(s);
        /* Many switches are read as "set or not": a false one stays unset. */
        if (s->type == T_BOOL && !v) {
            UNLOCK();
            return 0;
        }
        if (s->type == T_CHOICE && v >= 0 && v < s->n)
            snprintf(buf, sizeof buf, "%s", s->names[v]);
        else
            snprintf(buf, sizeof buf, "%d", v);
#if defined(_WIN32)
        _putenv_s(s->env, buf);
#endif
        done = 1;
    }
    UNLOCK();
    return done;
}

/* For the menu (ui_menu.cpp): walk the declared options of a section. */
int xbox_settings_enum(const char *section, int i, const char **key,
                       const char **label, int *type, int *lo, int *hi,
                       const char *const **names, int *n, unsigned *flags,
                       int *env_set)
{
    int k, seen = 0;
    LOCK();
    for (k = 0; k < s_count; k++) {
        Setting *s = &s_set[k];
        if (strcmp(s->section, section) || s->type == T_RAW || !s->label
            || (s->flags & XBOX_SETTING_HIDDEN))
            continue;
        if (seen++ == i) {
            *key = s->key;
            *label = s->label;
            *type = s->type;
            *lo = s->lo;
            *hi = s->hi;
            *names = s->names;
            *n = s->n;
            *flags = s->flags;
            *env_set = s->env && getenv(s->env) != NULL;
            UNLOCK();
            return 1;
        }
    }
    UNLOCK();
    return 0;
}
