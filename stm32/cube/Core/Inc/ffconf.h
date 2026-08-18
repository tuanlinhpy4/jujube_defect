/* ffconf.h — FatFs R0.12c config cho harness jujube.
 * Bật: STRFUNC (f_gets đọc manifest), LFN (tên file dài test_data.pack và .tflite),
 * ghi (results.csv + f_mkdir). 1 volume = thẻ SD. Xem PACK_FORMAT.md §1-3. */
#ifndef _FFCONF
#define _FFCONF 68300	/* R0.12c */

/*---- Function Configurations ----*/
#define _FS_READONLY    0
#define _FS_MINIMIZE    0      /* cần f_getfree/f_mkdir */
#define _USE_STRFUNC    1      /* f_gets */
#define _USE_FIND       0
#define _USE_MKFS       0
#define _USE_FASTSEEK   0
#define _USE_EXPAND     0
#define _USE_CHMOD      0
#define _USE_LABEL      0
#define _USE_FORWARD    0

/*---- Locale and Namespace ----*/
#define _CODE_PAGE      437    /* US */
#define _USE_LFN        1      /* 1 = bật, work-area tĩnh */
#define _MAX_LFN        255
#define _LFN_UNICODE    0
#define _STRF_ENCODE    3
#define _FS_RPATH       0

/*---- Drive/Volume Configurations ----*/
#define _VOLUMES        1
#define _STR_VOLUME_ID  0
#define _VOLUME_STRS    "RAM","NAND","CF","SD","SD2","USB","USB2","USB3"
#define _MULTI_PARTITION 0
#define _MIN_SS         512
#define _MAX_SS         512
#define _USE_TRIM       0
#define _FS_NOFSINFO    0

/*---- System Configurations ----*/
#define _FS_TINY        0
#define _FS_EXFAT       0
#define _FS_NORTC       1      /* không có RTC → timestamp cố định */
#define _NORTC_MON      1
#define _NORTC_MDAY     1
#define _NORTC_YEAR     2024
#define _FS_LOCK        0
#define _FS_REENTRANT   0
#define _FS_TIMEOUT     1000
#define _SYNC_t         HANDLE

#endif /* _FFCONF */
