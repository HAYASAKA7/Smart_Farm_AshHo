/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author         Notes
 * 2020-05-15     liuduanfei     first version
 */

#include "board.h"
#include "wm8988.h"
#include "drv_sound.h"
#include <rtthread.h>

//#define DRV_DEBUG

#define DBG_TAG              "drv.sound"
#ifdef  DRV_DEBUG
#define DBG_LVL              DBG_LOG
#else
#define DBG_LVL              DBG_INFO
#endif /* DRV_DEBUG */

#include <rtdbg.h>

#define CODEC_I2C_NAME  ("i2c3")

#define TX_DMA_FIFO_SIZE (2048)


static struct sai_reg * sai1 = (struct sai_reg *)SAI2BASE;

struct stm32_audio
{
    struct rt_i2c_bus_device *i2c_bus;
    struct rt_audio_device audio;
    struct rt_audio_configure replay_config;
    rt_int32_t replay_volume;
    rt_uint8_t *tx_fifo;
    rt_bool_t startup;
};
struct stm32_audio _stm32_audio_play = {0};

/* sample_rate, PLLI2SN(50.7), PLLI2SQ, PLLI2SDivQ, MCKDIV */
const rt_uint32_t SAI_PSC_TBL[][4] =
{
    {AUDIO_FREQUENCY_008K,256,5,25},
    {AUDIO_FREQUENCY_008K,302,107,0},
    {AUDIO_FREQUENCY_011K,426,2,52},
    {AUDIO_FREQUENCY_016K,429,38,2},
    {AUDIO_FREQUENCY_032K,426,1,52},
    {AUDIO_FREQUENCY_044K,429,1,38},
    {AUDIO_FREQUENCY_048K,467,1,38},
    {AUDIO_FREQUENCY_088K,429,1,19},
    {AUDIO_FREQUENCY_096K,467,1,19},
    {AUDIO_FREQUENCY_174K,271,1,6},
    {AUDIO_FREQUENCY_192K,295,6,0},
};

void SAIA_samplerate_set(rt_uint32_t freq)
{
    rt_uint16_t retry = 0;
    rt_uint8_t i = 0;
    rt_uint32_t temp = 0;
    for(i=0; i<(sizeof(SAI_PSC_TBL) / sizeof(SAI_PSC_TBL[0])); i++)
    {
        if(freq == SAI_PSC_TBL[i][0])break;
    }
    if(i == (sizeof(SAI_PSC_TBL) / sizeof(SAI_PSC_TBL[0])))
        return ;

    RCC->CR&=~(1<<26);
    while(((RCC->CR&(1<<27)))&&(retry<0X1FFF))retry++;
    RCC->PLLCKSELR &=~ (0X3F<<12);
    RCC->PLLCKSELR |= 25<<12;
    temp = RCC->PLL2DIVR;
    temp &=~ (0xFFFF);
    temp |= (SAI_PSC_TBL[i][1]-1)<<0;
    temp |= (SAI_PSC_TBL[i][2]-1)<<9;
    RCC->PLL2DIVR = temp;

    RCC->PLLCFGR |= 1<<19;
    RCC->CR |= 1<<26;
    while((RCC->CR&1<<27)==0);

    temp = sai1->acr1;
    temp &=~ (0X3F<<20);
    temp |= (rt_uint32_t)SAI_PSC_TBL[i][3]<<20;
    temp |= 1<<16;
    temp |= 1<<17;
    sai1->acr1 = temp;
}

void SAIA_channels_set(rt_uint16_t channels)
{
    if (channels == 2)
    {
        sai1->acr1 &= ~(1 << 12);
    }
    else
    {
        sai1->acr1 |= (1 << 12);
    }
}

void SAIA_samplebits_set(rt_uint16_t samplebits)
{
    LOG_D("%s",__func__);
    rt_uint32_t temp;
    switch (samplebits)
    {
    case 8:
        temp = 2;
        break;
    case 10:
        temp = 3;
        break;
    case 16:
        temp = 4;
        DMA2_Stream3->CR |= (1<<11);
        DMA2_Stream3->CR |= (1<<13);
        DMA2_Stream3->NDTR = TX_DMA_FIFO_SIZE / 4;
        break;
    case 20:
        temp = 5;
        break;
    case 24:
        temp = 6;
        DMA2_Stream3->CR &= ~(0xF<<11);
        DMA2_Stream3->CR |= (2<<11);
        DMA2_Stream3->CR |= (2<<13);
        DMA2_Stream3->NDTR = TX_DMA_FIFO_SIZE / 8;
        break;
    case 32:
        temp = 7;
        break;
    default:
        temp = 4;
        break;
    }
    sai1->acr1 &=~(7<<5);
    sai1->acr1 |= (temp<<5);
}

void SAIA_config_set(struct rt_audio_configure config)
{
    SAIA_channels_set(config.channels);
    SAIA_samplebits_set(config.samplebits);
    SAIA_samplerate_set(config.samplerate);
}

/* initial sai A */
rt_err_t SAIA_config_init()
{
    rt_uint32_t temp;
    /*
    Configuration register 1
      Master transmitter
      Free protocol
      Signals generated by the SAI change on SCK rising edge
      audio sub-block in asynchronous mode.
      Stereo mode
      Audio block output driven immediately after the setting of this bit.
      Master clock generator is enabled
    */
    temp = sai1->acr1;
    temp |= 1<<9;
    temp |= 1<<13;
    sai1->acr1 = temp;

    /*
    Frame configuration register
      Frame length 64
      Frame synchronization active level length 32
      FS signal is a start of frame signal + channel side identification
      FS is asserted one bit before the first bit of the slot 0
    */
    temp = (64-1)<<0;
    temp |= (32-1)<<8;
    temp |= 1<<16;
    temp |= 1<<18;
    sai1->afrcr = temp;

    /*
      Slot register
      Slot size 16
      Number of slots in an audio frame 2
      Slot0 Slot1 enable
    */
    temp = 1<<6;
    temp |= (2-1)<<8;
    temp |= (1<<17)|(1<<16);
    temp |= (1<<16);
    sai1->aslotr = temp;

    sai1->acr2 = 1<<0;
    sai1->acr2 |= 1<<3;

    return RT_EOK;
}

rt_err_t SAIA_tx_dma(void)
{
    RCC->AHB1ENR|=1<<1;
    RCC->D3AMR|=1<<0;
    while(DMA2_Stream3->CR&0X01);
    DMAMUX1_Channel11->CCR=89;

    DMA2->LIFCR|=0X3D<<22;
    DMA2_Stream3->FCR=0X0000021;

    DMA2_Stream3->CR=0;
    DMA2_Stream3->CR|=1<<6;
    DMA2_Stream3->CR|=1<<8;
    DMA2_Stream3->CR|=0<<9;
    DMA2_Stream3->CR|=1<<10;

    DMA2_Stream3->PAR  = (rt_uint32_t)&sai1->adr;
    DMA2_Stream3->M0AR = (rt_uint32_t)_stm32_audio_play.tx_fifo;
    DMA2_Stream3->M1AR = (rt_uint32_t)_stm32_audio_play.tx_fifo + (TX_DMA_FIFO_SIZE / 2);

    DMA2_Stream3->CR|=2<<16;
    DMA2_Stream3->CR|=1<<18;
    DMA2_Stream3->CR|=0<<21;
    DMA2_Stream3->CR|=0<<23;

    DMA2_Stream3->FCR&=~(1<<2);
    DMA2_Stream3->FCR&=~(3<<0);

    DMA2_Stream3->CR|=1<<4;

    HAL_NVIC_SetPriority(DMA2_Stream3_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream3_IRQn);
    return RT_EOK;
}

void DMA2_Stream3_IRQHandler(void)
{
    rt_interrupt_enter();
    if(DMA2->LISR&(1<<27))
    {
        DMA2->LIFCR|=1<<27;
        rt_audio_tx_complete(&_stm32_audio_play.audio);
    }
    rt_interrupt_leave();
}

rt_err_t sai_a_init()
{
    /* set sai_a DMA */
    SAIA_tx_dma();
    SAIA_config_init();
    return RT_EOK;
}

static rt_err_t stm32_player_getcaps(struct rt_audio_device *audio, struct rt_audio_caps *caps)
{
    rt_err_t result = RT_EOK;
    struct stm32_audio *st_audio = (struct stm32_audio *)audio->parent.user_data;

    LOG_D("%s:main_type: %d, sub_type: %d", __FUNCTION__, caps->main_type, caps->sub_type);

    switch (caps->main_type)
    {
    case AUDIO_TYPE_QUERY: /* query the types of hw_codec device */
    {
        switch (caps->sub_type)
        {
        case AUDIO_TYPE_QUERY:
            caps->udata.mask = AUDIO_TYPE_OUTPUT | AUDIO_TYPE_MIXER;
            break;

        default:
            result = -RT_ERROR;
            break;
        }

        break;
    }

    case AUDIO_TYPE_OUTPUT: /* Provide capabilities of OUTPUT unit */
    {
        switch (caps->sub_type)
        {
        case AUDIO_DSP_PARAM:
            caps->udata.config.channels     = st_audio->replay_config.channels;
            caps->udata.config.samplebits   = st_audio->replay_config.samplebits;
            caps->udata.config.samplerate   = st_audio->replay_config.samplerate;
            break;

        case AUDIO_DSP_SAMPLERATE:
            caps->udata.config.samplerate   = st_audio->replay_config.samplerate;
            break;

        case AUDIO_DSP_CHANNELS:
            caps->udata.config.channels     = st_audio->replay_config.channels;
            break;

        case AUDIO_DSP_SAMPLEBITS:
            caps->udata.config.samplebits     = st_audio->replay_config.samplebits;
            break;

        default:
            result = -RT_ERROR;
            break;
        }

        break;
    }

    case AUDIO_TYPE_MIXER: /* report the Mixer Units */
    {
        switch (caps->sub_type)
        {
        case AUDIO_MIXER_QUERY:
            caps->udata.mask = AUDIO_MIXER_VOLUME | AUDIO_MIXER_LINE;
            break;

        case AUDIO_MIXER_VOLUME:
            caps->udata.value = st_audio->replay_volume;
            break;

        case AUDIO_MIXER_LINE:
            break;

        default:
            result = -RT_ERROR;
            break;
        }

        break;
    }

    default:
        result = -RT_ERROR;
        break;
    }

    return result;
}

static rt_err_t stm32_player_configure(struct rt_audio_device *audio, struct rt_audio_caps *caps)
{
    rt_err_t result = RT_EOK;
    struct stm32_audio *st_audio = (struct stm32_audio *)audio->parent.user_data;

    LOG_D("%s:main_type: %d, sub_type: %d", __FUNCTION__, caps->main_type, caps->sub_type);

    switch (caps->main_type)
    {
    case AUDIO_TYPE_MIXER:
    {
        switch (caps->sub_type)
        {
        case AUDIO_MIXER_MUTE:
        {
            break;
        }

        case AUDIO_MIXER_VOLUME:
        {
            int volume = caps->udata.value;

            st_audio->replay_volume = volume;
            /* set mixer volume */
            wm8988_set_out_valume(_stm32_audio_play.i2c_bus, volume);
            break;
        }

        default:
            result = -RT_ERROR;
            break;
        }

        break;
    }

    case AUDIO_TYPE_OUTPUT:
    {
        switch (caps->sub_type)
        {
        case AUDIO_DSP_PARAM:
        {
            struct rt_audio_configure config = caps->udata.config;

            st_audio->replay_config.samplerate = config.samplerate;
            st_audio->replay_config.samplebits = config.samplebits;
            st_audio->replay_config.channels = config.channels;

            SAIA_config_set(config);
            break;
        }

        case AUDIO_DSP_SAMPLERATE:
        {
            st_audio->replay_config.samplerate = caps->udata.config.samplerate;
            SAIA_samplerate_set(caps->udata.config.samplerate);
            break;
        }

        case AUDIO_DSP_CHANNELS:
        {
            st_audio->replay_config.channels = caps->udata.config.channels;
            SAIA_channels_set(caps->udata.config.channels);
            break;
        }

        case AUDIO_DSP_SAMPLEBITS:
        {
            st_audio->replay_config.samplebits = caps->udata.config.samplebits;
            SAIA_samplebits_set(caps->udata.config.samplebits);
            break;
        }

        default:
            result = -RT_ERROR;
            break;
        }
        break;
    }

    default:
        break;
    }

    return result;
}

static rt_err_t stm32_player_init(struct rt_audio_device *audio)
{
    /* initialize wm8988 */
    _stm32_audio_play.i2c_bus = rt_i2c_bus_device_find(CODEC_I2C_NAME);

    sai_a_init();
    wm8988_init(_stm32_audio_play.i2c_bus);
    return RT_EOK;
}

static rt_err_t stm32_player_start(struct rt_audio_device *audio, int stream)
{
    if (stream == AUDIO_STREAM_REPLAY)
    {
        DMA2_Stream3->CR |= 1<<0;
    }

    return RT_EOK;
}

static rt_err_t stm32_player_stop(struct rt_audio_device *audio, int stream)
{
    if (stream == AUDIO_STREAM_REPLAY)
    {
        DMA2_Stream3->CR &= ~(1<<0);
    }

    return RT_EOK;
}

rt_size_t stm32_player_transmit(struct rt_audio_device *audio, const void *writeBuf, void *readBuf, rt_size_t size)
{
    SCB_CleanInvalidateDCache();
    return RT_EOK;
}

static void stm32_player_buffer_info(struct rt_audio_device *audio, struct rt_audio_buf_info *info)
{
    /**
     *               TX_FIFO
     * +----------------+----------------+
     * |     block1     |     block2     |
     * +----------------+----------------+
     *  \  block_size  / \  block_size  /
     */
    info->buffer = _stm32_audio_play.tx_fifo;
    info->total_size = TX_DMA_FIFO_SIZE;
    info->block_size = TX_DMA_FIFO_SIZE / 2;
    info->block_count = 2;
}
static struct rt_audio_ops _p_audio_ops =
{
    .getcaps     = stm32_player_getcaps,
    .configure   = stm32_player_configure,
    .init        = stm32_player_init,
    .start       = stm32_player_start,
    .stop        = stm32_player_stop,
    .transmit    = stm32_player_transmit,
    .buffer_info = stm32_player_buffer_info,
};

int rt_hw_sound_init(void)
{
    rt_uint8_t *tx_fifo;

    tx_fifo = rt_malloc(TX_DMA_FIFO_SIZE);
    if (tx_fifo == RT_NULL)
    {
        return -RT_ENOMEM;
    }
    rt_memset(tx_fifo, 0, TX_DMA_FIFO_SIZE);

    _stm32_audio_play.tx_fifo = tx_fifo;

    _stm32_audio_play.audio.ops = &_p_audio_ops;
    rt_audio_register(&_stm32_audio_play.audio, "sound0", RT_DEVICE_FLAG_WRONLY, &_stm32_audio_play);

    return RT_EOK;
}
INIT_DEVICE_EXPORT(rt_hw_sound_init);

int sai_pin(void)
{
    GPIO_InitTypeDef GPIO_Initure;

    RCC->D2CCIP1R &=~(0x7<<6);
    RCC->D2CCIP1R |=(0x1<<6);

    __HAL_RCC_SAI2_CLK_ENABLE();

    __HAL_RCC_GPIOI_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();

    /* PI4,7,5,6 */
    GPIO_Initure.Pin=GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7;
    GPIO_Initure.Mode=GPIO_MODE_AF_PP;
    GPIO_Initure.Pull=GPIO_PULLUP;
    GPIO_Initure.Speed=GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_Initure.Alternate=GPIO_AF10_SAI2;

    HAL_GPIO_Init(GPIOI,&GPIO_Initure);

    /* PG10 */
    GPIO_Initure.Pin=GPIO_PIN_10;
    HAL_GPIO_Init(GPIOG,&GPIO_Initure);

    return 0;
}
INIT_BOARD_EXPORT(sai_pin);
