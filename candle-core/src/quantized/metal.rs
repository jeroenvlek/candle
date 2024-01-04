use super::{GgmlDType, QStorage};
use crate::{DType, MetalDevice, MetalError, MetalStorage, Result};
use metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: GgmlDType) -> Self {
        Self {
            device,
            buffer,
            dtype,
        }
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        let buffer = self
            .device
            .new_buffer(elem_count, DType::F32, "dequantize")?;
        let device = &self.device;
        let command_buffer = device.command_buffer()?;
        let name = match self.dtype {
            GgmlDType::Q4_0 => "kernel_dequantize_q4_0",
            GgmlDType::Q4_1 => "kernel_dequantize_q4_1",
            GgmlDType::Q5_0 => "kernel_dequantize_q5_0",
            GgmlDType::Q5_1 => "kernel_dequantize_q5_1",
            GgmlDType::Q8_0 => "kernel_dequantize_q8_0",
            GgmlDType::Q8_1 => "kernel_dequantize_q8_1",
            GgmlDType::Q2K => "kernel_dequantize_q2_K",
            GgmlDType::Q3K => "kernel_dequantize_q3_K",
            GgmlDType::Q4K => "kernel_dequantize_q4_K",
            GgmlDType::Q5K => "kernel_dequantize_q5_K",
            GgmlDType::Q6K => "kernel_dequantize_q6_K",
            GgmlDType::Q8K => "kernel_dequantize_q8_K",
            GgmlDType::F16 => "kernel_dequantize_f16",
            GgmlDType::F32 => "kernel_dequantize_f32",
        };
        candle_metal_kernels::call_quantized_dequantize(
            device.device(),
            &command_buffer,
            device.kernels(),
            name,
            elem_count,
            &self.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(buffer, self.device.clone(), DType::F32))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }
}

pub fn load_quantized_metal<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}
