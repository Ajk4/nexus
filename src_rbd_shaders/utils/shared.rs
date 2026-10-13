//! Workgroup array access across CPU coroutine barriers.

#[inline(always)]
pub(crate) fn _read<T: Copy>(value: &T) -> T {
    #[cfg(not(target_arch_is_gpu))]
    unsafe {
        core::ptr::read_volatile(value)
    }
    #[cfg(target_arch_is_gpu)]
    {
        *value
    }
}

#[inline(always)]
pub(crate) fn _write<T: Copy>(value: &mut T, next: T) {
    #[cfg(not(target_arch_is_gpu))]
    unsafe {
        core::ptr::write_volatile(value, next);
    }
    #[cfg(target_arch_is_gpu)]
    {
        *value = next;
    }
}

pub trait WorkgroupArray<T: Copy> {
    fn _load(&self, index: usize) -> T;
    fn _store(&mut self, index: usize, value: T);
}

impl<T: Copy, const N: usize> WorkgroupArray<T> for [T; N] {
    #[inline(always)]
    fn _load(&self, index: usize) -> T {
        _read(&self[index])
    }

    #[inline(always)]
    fn _store(&mut self, index: usize, value: T) {
        _write(&mut self[index], value);
    }
}
