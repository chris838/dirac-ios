#define OFFSET_S16(ptr,offset) ((int16_t *)(((uint8_t *)(ptr)) + (offset)))
#define OFFSET_S32(ptr,offset) ((int32_t *)(((uint8_t *)(ptr)) + (offset)))
#define OFFSET_U8(ptr,offset) (((uint8_t *)(ptr)) + (offset))
#define OFFSET_U16(ptr,offset) ((uint16_t *)(((uint8_t *)(ptr)) + (offset)))
#define OFFSET_U32(ptr,offset) ((uint32_t *)(((uint8_t *)(ptr)) + (offset)))
#define ROUNDUP4(x) (((x)+3)&(~3))
#define ROUNDUP64(x) (((x)+63)&(~63))
struct __align__((4)) u8_4 { // 32 bit
    uint8_t a,b,c,d;
};

struct __align__((4)) i16_2 { // 32 bit
    int16_t a,b;
};

struct __align__((8)) i16_4 { // 64 bit
    int16_t a,b,c,d;
};
#define UINT(x) *((uint32_t*)&x)
