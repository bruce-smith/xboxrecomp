/* Folder helpers shared by netplay.c (which defines them) and netbag.c. */
#ifndef XBOX_NET_TREE_H
#define XBOX_NET_TREE_H

void np_tree_rm(const char *dir);                    /* the folder and all in it */
void np_tree_mk(const char *path);                   /* every level of path */
void np_tree_copy(const char *src, const char *dst); /* src's contents into dst */

#endif
